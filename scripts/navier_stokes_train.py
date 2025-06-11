# python script to train and test a FFM model on the navier stokes dataset
import sys
sys.path.append('../')

import gdown

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
from util.gaussian_process import GPPrior
from util.util import make_grid, reshape_for_batchwise, plot_loss_curve, plot_samples
import time

from util.util import load_navier_stokes
from torch.utils.data import TensorDataset, DataLoader

from models.fno import FNO
from models.FFM import FFMModel

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

downlaod_url = 'https://drive.google.com/uc?id=1rdaAAQqsz8vePMLd8iETgdk-35Twsc72'
data_identifier = 'data.npy'

ntr = 20000
batch_size = 256

modes = 64
visch = 1
hch = 64
pch = 64
xdim = 2
t_scaling = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'

lengthscale = 0.001
var = 1.0
sigma_min = 1e-4

lr = 1e-4

nepoch = 10
evalint = 2
saveint = 10
generate = True
spath = './models/'

def download_dataset(target_location='./'):
    print(f'Downloading dataset from {downlaod_url}')
    gdown.download(downlaod_url, target_location+data_identifier)
    print('Download complete!!!')

def load_and_prepare(data_location="./"):
    print(f'Loading data from {data_location+data_identifier}')
    data = torch.from_numpy(np.load(data_location+data_identifier))
    print(f'Loaded data with shape {data.shape}')
    data = data.permute(3, 1, 2, 0)
    data = data.permute(0, -1, 1, 2).contiguous().reshape(-1, 64, 64).unsqueeze(1) 
    idx = torch.randperm(data.shape[0])
    data = data[idx]
    print(f'Processed data to shape {data.shape}')
    data_tr, data_te = data[:ntr], data[ntr:]
    loader_tr = DataLoader(data_tr, batch_size=batch_size, shuffle=True)
    loader_te = DataLoader(data_te, batch_size=batch_size, shuffle=True)

    return loader_tr, loader_te

def create_model():
    print('Initializing model....')
    model = FNO(modes, visch, hch, pch, x_dim=xdim, t_scaling=t_scaling)
    model.to(device)
    print(model)
    model_wrapper = FFMModel(model, 
                         kernel_length=lengthscale, 
                         kernel_variance=var, 
                         sigma_min=sigma_min, 
                         device=device)
    return model, model_wrapper

# TODO: Run only if dataset is not present
# download_dataset()

loader_tr, loader_te = load_and_prepare()

model, model_wrapper = create_model()

optimizer = Adam(model.parameters(), lr)
scheduler = StepLR(optimizer, step_size=25, gamma=0.1)

print('Starting training....')

model_wrapper.train(loader_tr,
                    optimizer,
                    epochs=nepoch,
                    scheduler=scheduler,
                    test_loader=loader_te,
                    eval_int=evalint,
                    save_int=saveint,
                    generate=generate,
                    save_path=spath
                    )
import subprocess
import itertools
import torch

# Sweep space
models = ["fno", "unet", "deeponet"]
datasets = ["navier_stokes", "burgers"]
gpu_ids = list(range(torch.cuda.device_count()))

# Create (model, data) config combos
experiments = list(itertools.product(models, datasets))

print(f"[INFO] Launching {len(experiments)} experiments on {len(gpu_ids)} GPUs")

procs = []

for i, (model, data) in enumerate(experiments):
    gpu_id = gpu_ids[i % len(gpu_ids)]

    cmd = [
        "python", "train.py",
        f"model={model}",
        f"data={data}",
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        f"trainer.save_path=runs/{model}_{data}",  # optional
    ]

    print(f"[LAUNCH] GPU {gpu_id}: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    procs.append(proc)

# Wait for all to complete
for proc in procs:
    proc.wait()

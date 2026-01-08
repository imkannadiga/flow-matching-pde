# Flow Matching PDE

A PyTorch implementation for solving partial differential equations (PDEs) using flow matching techniques. This repository focuses on Navier-Stokes equations with support for multiple neural operator architectures including FNO (Fourier Neural Operator), LNO (Local Neural Operator), and U-Net.

## Features

- **Flow Matching**: Implements flow matching for continuous normalizing flows on PDEs
- **Multiple Architectures**: Supports FNO, LNO, and U-Net models
- **Navier-Stokes Dataset**: Pre-configured for 2D Navier-Stokes equations
- **Flexible Configuration**: Uses Hydra for configuration management
- **Experiment Tracking**: Optional Weights & Biases integration
- **Evaluation Metrics**: Includes MSE, density comparison, and spectrum analysis

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.7+

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd flow-matching-pde
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download data:
```bash
# Using Python script
python download_data.py

# Or using shell script (Linux/Mac)
./download_data.sh
```

**Note**: You may need to update `FILE_ID` or `DIRECT_URL` in the download scripts with the actual data source.

## Project Structure

```
flow-matching-pde/
├── configs/              # Hydra configuration files
│   ├── config.yaml      # Main config
│   ├── data/            # Data configurations
│   ├── model/           # Model configurations
│   ├── task/            # Task configurations
│   ├── train/           # Training configurations
│   └── eval/            # Evaluation configurations
├── data/                # Data modules
│   ├── base.py         # Base data module
│   ├── navier_stokes.py # NS data module
│   └── navier_stokes_fm.py # Flow matching NS data module
├── models/              # Model architectures
│   ├── fno.py          # Fourier Neural Operator
│   ├── lno.py          # Local Neural Operator
│   └── unet.py         # U-Net
├── tasks/               # Task implementations
│   ├── base.py         # Base task class
│   ├── fm-ns.py        # Flow matching Navier-Stokes task
│   ├── fno-ns.py       # FNO Navier-Stokes task
│   └── lno-ns.py       # LNO Navier-Stokes task
├── training/            # Training utilities
│   ├── trainer.py      # Training loop
│   ├── loss.py         # Loss functions
│   └── data_processors.py # Data preprocessing
├── evaluation/          # Evaluation utilities
│   ├── base.py         # Base evaluator
│   ├── ns_evaluator.py # Navier-Stokes evaluator
│   └── eval_utils.py   # Evaluation helpers
├── train.py            # Training entry point
├── eval.py             # Evaluation entry point
└── start_tasks.py      # Task runner entry point
```

## Usage

### Training

#### Using the standard trainer (FNO/LNO):
```bash
python train.py \
    model=fno \
    data=navier_stokes \
    trainer=run \
    wandb=disabled
```

#### Using the flow matching task:
```bash
python start_tasks.py \
    task=fm-ns \
    model=unet \
    data=navier_stokes_fm \
    wandb=disabled
```

### Evaluation

```bash
python eval.py \
    model=fno \
    data=navier_stokes \
    eval=simulation \
    eval.state_dict_path=training_runs/fno-navier_stokes-trial/checkpoints
```

### Configuration

All configurations are managed through Hydra. Key configuration files:

- `configs/config.yaml`: Main configuration file
- `configs/train/base.yaml`: Training hyperparameters
- `configs/model/`: Model-specific configurations
- `configs/data/`: Dataset configurations

You can override any configuration via command line:
```bash
python train.py model=fno model.modes=32 train.epochs=100 train.lr=0.001
```

## Models

### FNO (Fourier Neural Operator)
- **File**: `models/fno.py`
- **Forward signature**: `forward(t, u)`
- **Key parameters**: `modes`, `hidden_channels`, `proj_channels`

### LNO (Local Neural Operator)
- **File**: `models/lno.py`
- **Forward signature**: `forward(t, u)`
- **Key parameters**: `modes`, `hidden_channels`, `disco_kernel_shape`

### U-Net
- **File**: `models/unet.py`
- **Forward signature**: `forward(t, u)`
- **Key parameters**: `in_channels`, `out_channels`, `base_channels`

## Data

The Navier-Stokes dataset consists of:
- **Shape**: [T, H, W, N] where:
  - T = 50 (time steps)
  - H = W = 64 (spatial resolution)
  - N = 5000 (number of trajectories)
  - Re = 20 (Reynolds number)

Data is automatically downloaded and preprocessed. The `BaseDataModule` handles:
- Loading and preprocessing
- Train/test splits
- DataLoader creation

## Evaluation Metrics

The evaluation module provides:
- **MSE per step**: Mean squared error at each time step
- **Density MSE**: KDE-based distribution comparison
- **Spectrum comparison**: Energy spectrum analysis
- **Visualization**: Sequence plots and comparisons

## Weights & Biases Integration

To enable W&B logging:

1. Install wandb: `pip install wandb`
2. Configure in `configs/wandb/enabled.yaml`
3. Use: `python train.py wandb=enabled`

Or override via command line:
```bash
python train.py wandb=enabled wandb.project=my_project wandb.entity=my_entity
```

## Troubleshooting

### Missing Data
If you encounter data loading errors:
1. Ensure data files are in `data/_data/navier_stokes/`
2. Run `python download_data.py` to download data
3. Update `FILE_ID` or `DIRECT_URL` in download scripts if needed

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use gradient accumulation
- Enable mixed precision training

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify `neuralop` package is installed: `pip install neuralop`

## Configuration Reference

### Training Configuration
```yaml
lr: 0.0001              # Learning rate
epochs: 200             # Number of epochs
eval_int: 10            # Evaluation interval
save_int: 20            # Checkpoint save interval
device: cuda            # Device (cuda/cpu)
dt: 0.02                # Time step (for flow matching)
```

### Model Configuration
```yaml
modes: 64               # Fourier modes (FNO/LNO)
hidden_channels: 64     # Hidden channels
proj_channels: 64       # Projection channels
t_scaling: 1000         # Time scaling factor
```

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:
```bibtex
[Add citation information]
```

## Contributing

[Add contributing guidelines if applicable]

## Acknowledgments

- Neural Operators library: [neuraloperator](https://github.com/neuraloperator/neuraloperator)
- Hydra framework: [hydra](https://hydra.cc/)

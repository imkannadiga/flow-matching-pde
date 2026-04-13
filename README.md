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
│   ├── navier_stokes_fm.py # Flow matching NS data module
│   └── darcy.py # Darcy flow HDF5 (nu → pressure)
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
├── evaluate.py         # Rollout benchmark (Hydra ``evaluate`` config)
├── eval.py             # Legacy NSEvaluator entry point
└── start_tasks.py      # Task runner entry point
```

## Usage

### Training

#### Using `train.py` (Hydra)

Set `PROJECT_ROOT` to the repository root so data paths resolve. Each run writes `config_hash.txt` and `results.json` under the Hydra output directory, and uses trajectory-level train/test splits (no leakage across trajectories).

Training is `accelerate`-native by default, including single-node single-GPU runs.

**CUDA OOM / large batches:** keep `data.batch_size` at a size that fits in memory, and increase ``trainer.gradient_accumulation_steps`` so each optimizer step uses gradients summed over that many microbatches (effective batch ≈ ``batch_size × gradient_accumulation_steps`` with the default sum-reduction MSE). Optionally set ``trainer.mixed_precision=true`` (autocast) to reduce memory further.

```bash
export PROJECT_ROOT=$(pwd)
python train.py \
    model=fno \
    data=navier_stokes \
    trainer=run \
    wandb=disabled
```

### Accelerate launch matrix

`data.batch_size` is per-process (per GPU). Effective update batch is:

`data.batch_size * accelerate.num_processes * trainer.gradient_accumulation_steps`

Single GPU (default config path):

```bash
accelerate launch --num_processes 1 train.py trainer=run accelerate=single_gpu
```

Single node, multi-GPU:

```bash
accelerate launch --num_processes 4 train.py trainer=run accelerate=multi_gpu accelerate.num_processes=4
```

Multi-node, multi-GPU (direct `accelerate launch` on each node):

```bash
# Node 0
accelerate launch \
  --num_machines 2 \
  --machine_rank 0 \
  --num_processes 4 \
  --main_process_ip 10.0.0.1 \
  --main_process_port 29500 \
  train.py trainer=run accelerate=multi_node accelerate.machine_rank=0

# Node 1
accelerate launch \
  --num_machines 2 \
  --machine_rank 1 \
  --num_processes 4 \
  --main_process_ip 10.0.0.1 \
  --main_process_port 29500 \
  train.py trainer=run accelerate=multi_node accelerate.machine_rank=1
```

Multi-node with node-rank launcher (`launch_train.py`):

```bash
# Same command template on each node; only --node-id changes.
python launch_train.py \
  --node-id 0 \
  --num-machines 2 \
  --num-processes 4 \
  --main-process-ip 10.0.0.1 \
  --main-process-port 29500 \
  -- trainer=run accelerate=multi_node

python launch_train.py \
  --node-id 1 \
  --num-machines 2 \
  --num-processes 4 \
  --main-process-ip 10.0.0.1 \
  --main-process-port 29500 \
  -- trainer=run accelerate=multi_node
```

Optional: `torch-harmonics` is required for **LFNO** (`model=lno`) with the current `neuralop` DISCO stack; FNO and U-Net do not need it.

#### Paradigms (FM / AR / diffusion)

Training uses ``configs/paradigm/{fm,ar,diffusion}.yaml`` to set the processor (rectified **bridge** flow, autoregressive one-step, or VP diffusion with ε-prediction). Switch with e.g. ``paradigm=ar``.

Phase 1 multirun (paradigm × seed):

```bash
python train.py --config-path configs --config-name experiment/phase1 --multirun
```

#### Conditioning (levels 0 / 1 / 2)

Training defaults include ``conditioning: level0`` (see ``configs/conditioning/``). Level 1 concatenates a normalized spatial grid to ``u`` before the model; level 2 also threads ``batch["params"]`` into the forward pass for FiLM. For level 1 or 2, set ``model.coord_channels=2`` for FNO, LNO, U-Net, ViT, and AM-FNO so channel counts match the processor. For level 2 on NumPy Navier-Stokes without per-trajectory metadata, use ``data.default_param_vector`` (values in the same order as ``conditioning.param_keys``, default ``[Re, nu]``).

Flow matching and diffusion processors (``training/paradigms/fm.py``, ``diffusion.py``) build the core path on ``batch["x"]`` and ``batch["y"]`` (same channel count for the rectified bridge), then **stack** optional tensors onto ``u``: ``batch["x_aux"]``, per-channel ``channel_mins`` / ``channel_maxs`` (each ``(B, C_field)``), and expanded ``batch["params"]``. Toggle stacking via ``paradigm.pre_train_processor.stack_*``. **Input/output widths for the backbone are not set in YAML:** ``training/model_channels.py`` patches ``cfg.model`` from the first preprocessed batch—``out_channels`` is always ``batch["y"].shape[1]`` (the supervised field), and the input width matches ``batch["x"]["u"].shape[1]`` (with the usual ``coord_channels`` split for FNO / ViT / U-Net). U-Net uses ``in_channels`` / ``out_channels`` in the same way internally.

Phase 2 (FM + level 2 + synthetic constant params) and phase 3 (architecture × seed sweep under full conditioning):

```bash
python train.py --config-path configs --config-name experiment/phase2 --multirun
python train.py --config-path configs --config-name experiment/phase3 --multirun
```

Compose / instantiate smoke (no data I/O beyond config): ``python scripts/dry_compose_sprint3.py``. PDEBench-style HDF5 loads are configured via ``configs/data/pdebench_ns.yaml`` and ``data/pdebench_ns.py``. Darcy-flow HDF5 pairs (``nu`` / ``tensor``) use ``configs/data/darcy.yaml`` and ``data/darcy.py``; set ``load_all_hdf5=true`` to merge every ``*.hdf5`` / ``*.h5`` in ``data_dir``, with per-file ``beta`` from the filename in ``batch["params"]`` and optionally as an extra constant channel on ``x``. ``BaseDataModule.load_hdf5_arrays`` supports reading named datasets from a single file.

Optional hardware presets: ``defaults`` entry ``hardware=a100`` or ``hardware=4090`` (overrides ``data.batch_size``; see ``configs/hardware/``).

#### Using the flow matching task:
```bash
python start_tasks.py \
    task=fm-ns \
    model=unet \
    data=navier_stokes_fm \
    wandb=disabled
```

### Evaluation

**Rollout benchmark (JSONL row + metrics):** ``evaluate.py`` runs ``RolloutEvaluator`` (trajectory rollouts, timing, spectrum/density). Point it at a training checkpoint directory (with ``model_state_dict.pt`` or ``manifest.pt``):

```bash
export PROJECT_ROOT=$(pwd)
export CHECKPOINT_DIR=/path/to/your/hydra/run/checkpoints
python evaluate.py paradigm=fm evaluate.sampler_type=euler
```

Each run appends a line to ``results.jsonl`` in the Hydra output directory. To merge JSONL files into ``results.csv``, use ``evaluation.results_logger.export_csv`` from Python (see module docstring).

With the current model inputs, the **diffusion** paradigm’s DDIM path samples a next state from Gaussian noise only (it does not condition on ``u_t``); FM (bridge) and AR samplers use the current field as in training.

**Legacy visualization / single-step checks:** ``eval.py`` with ``NSEvaluator`` and ``configs/eval.yaml``:

```bash
python eval.py \
    model=fno \
    data=navier_stokes \
    eval=base \
    train.save_path=training_runs/fno-navier_stokes-trial/checkpoints
```

`eval.py` loads weights from `eval.state_dict_path` when set, otherwise `train.save_path`.

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
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `modes`, `hidden_channels`, `proj_channels`, `t_scaling` (default `1`)

### LNO (Local Neural Operator)
- **File**: `models/lno.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `modes`, `hidden_channels`, `disco_kernel_shape`

### U-Net
- **File**: `models/unet.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `in_channels`, `out_channels`, `base_channels`, `coord_channels` (optional extra field channels when coords are pre-concatenated into `u`), `film_param_dim` for FiLM blocks

### Field ViT
- **File**: `models/vit.py` (`FieldViT`)
- **Forward signature**: `forward(t, u, coords=None, params=None)` (``coords`` unused; coords should be concatenated into ``u`` when ``coord_channels>0``)
- **Key parameters**: `patch_size`, `embed_dim`, `depth`, `num_heads`, `coord_channels`, `film_param_dim`

### AM-FNO
- **File**: `models/amfno.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `param_dim`, `context_dim` (parameter MLP → tiled context maps before the spectral trunk), `coord_channels`, `film_param_dim`

## Data

The Navier-Stokes dataset consists of:
- **Shape**: [T, H, W, N] where:
  - T = 50 (time steps)
  - H = W = 64 (spatial resolution)
  - N = 5000 (number of trajectories)
  - Re = 20 (Reynolds number)

Data is automatically downloaded and preprocessed. The `BaseDataModule` handles:
- Loading and preprocessing to **(N, T, C, H, W)**
- **Trajectory-level** train/test splits (pairs are drawn only within each split’s trajectories)
- Optional multi-file load when `filename` is omitted (glob of `*.npy` / `*.pt` in `data_dir`)
- DataLoader creation

`NSDataModule.domain_bounds` documents the physical domain (default unit square `(0,1)^2`).

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

# Flow Matching PDE

A PyTorch implementation for solving PDE surrogates with a flow-matching style training pipeline. Supports Darcy flow and Shallow Water Equations (SWE), with multiple model backbones including U-Net, FNO, LNO, ViT, and AM-FNO.

## Features

- **Flow-Matching Processor**: Uses `FlowMatchingProcessor` in the default trainer config
- **Multiple Architectures**: Supports U-Net, FNO, LNO, ViT, and AM-FNO
- **Multiple Datasets**: Darcy flow HDF5 and Shallow Water Equations (SWE) with auto-regressive rollout
- **Flexible Configuration**: Uses Hydra for configuration management
- **Accelerate by Default**: Single-GPU, multi-GPU, and multi-node launches share one codepath
- **Experiment Tracking**: Optional Weights & Biases integration
- **Checkpointing**: Saves model/optimizer/scheduler manifests for resume
- **Evaluation Pipeline**: ODE solvers (Euler, RK4, Dopri5) with per-step metrics and sample storage

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

3. Place your HDF5 data files at the paths configured in `configs/data/`:
   - Darcy: `data/_data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5`
   - SWE: `data/_data/shallow-water/2D_rdb_NA_NA.h5`

   Override the path at runtime if needed: `python train.py data.data_path=/abs/path/to/file.hdf5`

## Project Structure

```text
flow-matching-pde/
├── configs/
│   ├── config.yaml               # Main Hydra defaults (training)
│   ├── eval.yaml                 # Main Hydra defaults (evaluation)
│   ├── accelerate/               # Accelerate runtime profiles
│   ├── data/                     # Dataset configs (darcy.yaml, swe.yaml)
│   ├── evaluator/                # Evaluator + solver configs (euler, rk4, dopri5)
│   ├── model/                    # Model configs
│   ├── trainer/                  # Trainer + optimizer/scheduler/loss configs
│   ├── train/                    # Extra train presets
│   └── wandb/                    # W&B on/off configs
├── data/
│   ├── base.py
│   ├── darcy.py
│   └── swe.py
├── evaluation/
│   ├── evaluator.py              # FlowMatchingEvaluator (one-step + rollout)
│   ├── metrics.py                # RelativeL2, MSE, Linf trackers
│   ├── packers.py                # ConditionPacker
│   ├── sample_store.py           # Optional prediction/target storage
│   └── solvers.py                # EulerSolver, RK4Solver, Dopri5Solver
├── models/
│   ├── unet.py
│   ├── fno.py
│   ├── lno.py
│   ├── vit.py
│   └── amfno.py
├── training/
│   ├── trainer.py
│   ├── data_processors.py
│   └── training_state.py
├── util/                         # Shared utilities (config, reproducibility, etc.)
├── train.py                      # Main Hydra training entrypoint
├── evaluate.py                   # Main Hydra evaluation entrypoint
├── launch_train.py               # Node-rank launcher wrapper for multi-node training
├── launch_eval.py                # Node-rank launcher wrapper for multi-node evaluation
└── visualize_swe_eval.py         # Visualization helper for SWE rollout results
```

## Usage

### Training

Each run writes `metrics.json` and `resolved_config.yaml` under the Hydra output directory (`runs/<data>-<model>-<timestamp>/`).

Training is `accelerate`-native by default, including single-node single-GPU runs.

**CUDA OOM / large batches:** keep `data.batch_size` at a size that fits in memory, and increase `trainer.gradient_accumulation_steps` so each optimizer step uses gradients summed over that many microbatches (effective batch ≈ `batch_size × gradient_accumulation_steps` with the default sum-reduction MSE). Optionally set `accelerate.mixed_precision=fp16` to reduce memory further.

```bash
# Default: Darcy + U-Net
python train.py model=unet data=darcy trainer=run wandb=disabled

# Shallow Water Equations + FNO
python train.py model=fno data=swe trainer=run wandb=disabled
```

### Accelerate launch matrix

`data.batch_size` is per-process (per GPU). Effective update batch is:

`data.batch_size * accelerate.num_processes * trainer.gradient_accumulation_steps`

Single GPU:

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
  --num_machines 2 --machine_rank 0 \
  --num_processes 4 \
  --main_process_ip 10.0.0.1 --main_process_port 29500 \
  train.py trainer=run accelerate=multi_node accelerate.machine_rank=0

# Node 1
accelerate launch \
  --num_machines 2 --machine_rank 1 \
  --num_processes 4 \
  --main_process_ip 10.0.0.1 --main_process_port 29500 \
  train.py trainer=run accelerate=multi_node accelerate.machine_rank=1
```

Multi-node with node-rank launcher (`launch_train.py`):

```bash
# Same command on each node; only --node-id changes
python launch_train.py \
  --node-id 0 --num-machines 2 --num-processes 4 \
  --main-process-ip 10.0.0.1 --main-process-port 29500 \
  -- trainer=run accelerate=multi_node

python launch_train.py \
  --node-id 1 --num-machines 2 --num-processes 4 \
  --main-process-ip 10.0.0.1 --main-process-port 29500 \
  -- trainer=run accelerate=multi_node
```

### Common Hydra overrides

```bash
python train.py --help  # show available groups/options

# Switch model
python train.py model=fno trainer=debug

# Enable W&B
python train.py wandb=enabled wandb.project=my_project wandb.entity=my_entity

# Override batch size and optimizer knobs
python train.py data.batch_size=16 trainer.gradient_accumulation_steps=2 accelerate.mixed_precision=fp16

# Full override example
python train.py model=fno data=swe trainer.n_epochs=100 trainer.optimizer.lr=0.001
```

---

### Evaluation

Evaluation uses a separate Hydra config (`configs/eval.yaml`) and writes `eval_metrics.json` and `resolved_config.yaml` under `evals/<data>-<model>-<timestamp>/`.

You must supply a `checkpoint_dir` pointing to the checkpoint directory created during training.

#### One-step evaluation (Darcy and similar)

```bash
python evaluate.py \
  model=unet data=darcy evaluator=euler \
  checkpoint_dir=runs/darcy-unet-2024-01-01_12-00-00/checkpoints
```

#### Auto-regressive rollout evaluation (SWE)

Set `time_variate=true` — the evaluator automatically activates dataset eval mode and runs full-trajectory rollout:

```bash
python evaluate.py \
  model=fno data=swe evaluator=rk4 \
  checkpoint_dir=runs/swe-fno-2024-01-01_12-00-00/checkpoints \
  data.time_variate=true
```

#### Changing the ODE solver

```bash
# Euler (default, fast)
python evaluate.py evaluator=euler checkpoint_dir=...

# Runge-Kutta 4
python evaluate.py evaluator=rk4 checkpoint_dir=...

# Dormand-Prince (adaptive, via torchdiffeq)
python evaluate.py evaluator=dopri5 checkpoint_dir=...
```

#### Multi-node evaluation (`launch_eval.py`)

```bash
python launch_eval.py \
  --node-id 0 --num-machines 2 --num-processes 4 \
  --main-process-ip 10.0.0.1 --main-process-port 29500 \
  -- checkpoint_dir=runs/... evaluator=euler
```

---

## Models

### FNO (Fourier Neural Operator)
- **File**: `models/fno.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `modes`, `hidden_channels`, `proj_channels`

### LNO (Local Neural Operator)
- **File**: `models/lno.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `modes`, `hidden_channels`, `disco_kernel_shape`

### U-Net
- **File**: `models/unet.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `in_channels`, `out_channels`, `base_channels`, `coord_channels`, `film_param_dim`

### Field ViT
- **File**: `models/vit.py` (`FieldViT`)
- **Forward signature**: `forward(t, u, coords=None, params=None)` (coords should be concatenated into `u` when `coord_channels > 0`)
- **Key parameters**: `patch_size`, `embed_dim`, `depth`, `num_heads`, `coord_channels`, `film_param_dim`

### AM-FNO
- **File**: `models/amfno.py`
- **Forward signature**: `forward(t, u, coords=None, params=None)`
- **Key parameters**: `param_dim`, `context_dim`, `coord_channels`, `film_param_dim`

---

## Data

### Darcy Flow
- **Input**: `nu`, **Target**: `tensor`
- **Default path**: `data/_data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5`
- **Config**: `configs/data/darcy.yaml` / `data/darcy.py`

### Shallow Water Equations (SWE)
- **Layout**: per-trajectory HDF5 keys `XXXX/data` `(T, H, W, 1)`, `XXXX/grid/t`, `XXXX/grid/x`, `XXXX/grid/y`
- **Default path**: `data/_data/shallow-water/2D_rdb_NA_NA.h5`
- **Config**: `configs/data/swe.yaml` / `data/swe.py`
- **Training mode**: produces `N × (T-1)` one-step pairs per trajectory
- **Eval mode**: returns full trajectories for auto-regressive rollout
- **Notable options** (in `configs/data/swe.yaml`):
  - `append_physical_time`: broadcast the PDE time as a spatial channel
  - `append_coords`: append x/y coordinate maps as conditioning channels
  - `preload`: load all trajectories into RAM (for small datasets)

---

## Weights & Biases Integration

```bash
pip install wandb
python train.py wandb=enabled wandb.project=my_project wandb.entity=my_entity
```

---

## Troubleshooting

### Missing Data
- Ensure the HDF5 file exists at `data.data_path`
- Override at runtime: `python train.py data.data_path=/abs/path/to/file.hdf5`

### CUDA Out of Memory
- Reduce `data.batch_size`
- Increase `trainer.gradient_accumulation_steps`
- Enable mixed precision: `accelerate.mixed_precision=fp16`

### Import Errors
- Install all dependencies: `pip install -r requirements.txt`
- Verify `neuralop` is installed: `pip install neuraloperator`
- For adaptive ODE solvers: `pip install torchdiffeq`

---

## Notes

- Checkpoints are saved under `${hydra.run.dir}/checkpoints` — files: `model_state_dict.pt`, `optimizer.pt`, `scheduler.pt`, `manifest.pt`.
- Logging side effects (stdout/tqdm/W&B) are restricted to the main process in distributed jobs.
- Physical PDE time and flow-matching time `τ ∈ [0, 1]` are distinct; the SWE dataset can optionally append the former as a conditioning channel so the model knows its position in the physical trajectory.

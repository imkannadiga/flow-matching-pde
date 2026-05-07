# Flow Matching PDE

A PyTorch implementation for solving PDE surrogates with a flow-matching style training pipeline. The current default dataset is Darcy flow HDF5 data, with multiple model backbones including U-Net, FNO, LNO, ViT, and AM-FNO.

## Features

- **Flow-Matching Processor**: Uses `FlowMatchingProcessor` in the default trainer config
- **Multiple Architectures**: Supports U-Net, FNO, LNO, ViT, and AM-FNO
- **Darcy Dataset**: Pre-configured for Darcy HDF5 (`nu` -> `tensor`) training
- **Flexible Configuration**: Uses Hydra for configuration management
- **Accelerate by Default**: Single-GPU, multi-GPU, and multi-node launches share one codepath
- **Experiment Tracking**: Optional Weights & Biases integration
- **Checkpointing**: Saves model/optimizer/scheduler manifests for resume

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

```text
flow-matching-pde/
├── configs/
│   ├── config.yaml               # Main Hydra defaults
│   ├── accelerate/               # accelerate runtime profiles
│   ├── data/                     # Dataset config(s)
│   ├── model/                    # Model config(s)
│   ├── trainer/                  # Trainer + optimizer/scheduler/loss configs
│   ├── train/                    # Extra train presets
│   └── wandb/                    # W&B on/off configs
├── data/
│   ├── base.py
│   └── darcy.py
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
├── train.py                      # Main Hydra training entrypoint
└── launch_train.py               # Node-rank launcher wrapper for multi-node
```

## Usage

### Training

#### Using `train.py` (Hydra)

Each run writes `metrics.json` and `resolved_config.yaml` under the Hydra output directory.

Training is `accelerate`-native by default, including single-node single-GPU runs.

**CUDA OOM / large batches:** keep `data.batch_size` at a size that fits in memory, and increase ``trainer.gradient_accumulation_steps`` so each optimizer step uses gradients summed over that many microbatches (effective batch ≈ ``batch_size × gradient_accumulation_steps`` with the default sum-reduction MSE). Optionally set ``trainer.mixed_precision=true`` (autocast) to reduce memory further.

```bash
python train.py model=unet data=darcy trainer=run wandb=disabled
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

### Common Hydra overrides

Show available groups/options:

```bash
python train.py --help
```

Examples:

```bash
# Switch model and trainer preset
python train.py model=fno trainer=debug

# Enable W&B
python train.py wandb=enabled

# Override runtime and optimization knobs
python train.py data.batch_size=16 trainer.gradient_accumulation_steps=2 accelerate.mixed_precision=fp16
```

### Configuration

All configurations are managed through Hydra. Key files:

- `configs/config.yaml`: Main configuration file
- `configs/accelerate/*.yaml`: Runtime/distributed settings
- `configs/model/`: Model-specific configurations
- `configs/data/`: Dataset configurations
- `configs/trainer/`: Trainer/optimizer/scheduler/loss wiring

You can override any configuration via command line:
```bash
python train.py model=fno trainer.n_epochs=100 trainer.optimizer.lr=0.001
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

The default dataset is Darcy flow from HDF5:

- input: `nu`
- target: `tensor`
- default path: `data/_data/darcy/2D_DarcyFlow_beta1.0_Train.hdf5`

Configured in `configs/data/darcy.yaml` and loaded by `data/darcy.py`.

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
1. Ensure the configured file exists at `data.data_path` (default in `configs/data/darcy.yaml`)
2. Override the data path explicitly if needed, e.g. `python train.py data.data_path=/abs/path/to/file.hdf5`

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use gradient accumulation
- Enable mixed precision training

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Verify `neuralop` package is installed: `pip install neuralop`

## Notes

- Checkpoints are saved under `${hydra.run.dir}/checkpoints` (`model_state_dict.pt`, `optimizer.pt`, `scheduler.pt`, `manifest.pt`).
- Logging side effects (stdout/tqdm/W&B) are restricted to the main process when running distributed jobs.

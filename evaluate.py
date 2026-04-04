"""
Hydra entry for rollout evaluation (JSONL row + stdout metrics).

Example::

  python evaluate.py evaluate.checkpoint_dir=/path/to/hydra/run/checkpoints paradigm=ar
"""

import hydra
from omegaconf import DictConfig

from evaluation.rollout_eval import RolloutEvaluator


@hydra.main(config_path="configs", config_name="evaluate", version_base=None)
def main(cfg: DictConfig):
    RolloutEvaluator(cfg).run()


if __name__ == "__main__":
    main()

"""
Legacy entry: ``python eval.py`` uses ``configs/eval.yaml`` and ``NSEvaluator``.

For rollout JSONL + CSV aggregation (Sprint 2 benchmark), prefer ``python evaluate.py``
with ``configs/evaluate.yaml``.
"""

import hydra

import models  # noqa: F401 — applies torch.empty / tltorch compat before neuralop loads

from omegaconf import DictConfig

from evaluation.ns_evaluator import NSEvaluator


@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    """Hydra entry: ``NSEvaluator`` loads model and data from ``cfg``."""
    evaluator = NSEvaluator(cfg=cfg)
    evaluator.run()


if __name__ == "__main__":
    main()

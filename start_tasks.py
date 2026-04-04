import hydra

import models  # noqa: F401 — applies torch.empty / tltorch compat before neuralop loads

from omegaconf import DictConfig
from hydra.utils import instantiate
from evaluation.ns_evaluator import NSEvaluator
from omegaconf import OmegaConf
from util.reproducibility import wandb_run_name

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Optional wandb init
    if cfg.task.do_train:
        if cfg.get("wandb", {}).get("use_wandb", False):
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=wandb_run_name(cfg),
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                mode=cfg.wandb.mode,
            )

        # Instantiate and run the task
        task = instantiate(cfg.task)
        task.run()
        
        if cfg.get("wandb", {}).get("use_wandb", False):
            wandb.finish()
    
    if cfg.task.do_evaluate:
        # Run evaluation by passing the entire config
        evaluator = NSEvaluator(cfg=cfg)
        evaluator.run()

if __name__ == "__main__":
    main()

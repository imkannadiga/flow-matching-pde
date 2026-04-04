import hydra

import models  # noqa: F401 — applies torch.empty / tltorch compat before neuralop loads

from omegaconf import DictConfig
from hydra.utils import instantiate
from evaluation.ns_evaluator import NSEvaluator
from omegaconf import OmegaConf
from util.reproducibility import wandb_group, wandb_run_id, wandb_run_name

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Optional wandb init
    if cfg.task.do_train:
        if cfg.get("wandb", {}).get("use_wandb", False):
            import wandb
            wb_kwargs = dict(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                id=wandb_run_id(cfg),
                name=wandb_run_name(cfg),
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                mode=cfg.wandb.mode,
                resume="never",
            )
            _grp = wandb_group(cfg)
            if _grp is not None:
                wb_kwargs["group"] = _grp
            wandb.init(**wb_kwargs)

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

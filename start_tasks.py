import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
from evaluation.eval import evaluate

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Optional wandb init
    if cfg.task.do_train:
        if cfg.get("wandb", {}).get("use_wandb", False):
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=cfg,
                name=cfg.wandb.run_name,
                mode=cfg.wandb.mode,
            )

        # Instantiate and run the task
        task = instantiate(cfg.task)
        task.run()
        
        if cfg.get("wandb", {}).get("use_wandb", False):
            wandb.finish()
    
    if cfg.task.do_evaluate:
        # Run evaluation by passing the entire config
        evaluate(cfg)

if __name__ == "__main__":
    main()

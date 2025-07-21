import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import torch
import os

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    
    data = instantiate(cfg.data)
    model = instantiate(cfg.model)
    train_loader, test_loader = data.get_dataloaders()

    # Instantiate the components
    pre_train_processor = instantiate(cfg.trainer.pre_train_processor)
    
    _optimizer = instantiate(cfg.trainer.optimizer)
    optimizer = _optimizer(params=model.parameters())
    
    _scheduler = instantiate(cfg.trainer.scheduler)
    scheduler = _scheduler(optimizer=optimizer)
    
    loss_fn = instantiate(cfg.trainer.loss)
    regularizer = instantiate(cfg.trainer.regularizer) if 'regularizer' in cfg.trainer else None

    if cfg.get("wandb", {}).get("use_wandb", False):
        import wandb
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.model.name,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            mode=cfg.wandb.mode,
        )


    trainer = instantiate(cfg.trainer, model=model)

    trainer.train(
        train_loader=train_loader,
        test_loaders={"64x64": test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=regularizer,
        training_loss=loss_fn,
        eval_losses={"mse":loss_fn},  # Can extend to support multiple
        save_every=cfg.trainer.save_every,
        save_dir=cfg.trainer.save_path,
        resume_from_dir=cfg.trainer.resume_from_dir if 'resume_from_dir' in cfg.trainer else None,
    )
    
    if cfg.get("wandb", {}).get("use_wandb", False):
        wandb.finish()
    
if __name__ == "__main__":
    main()

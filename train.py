import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    
    data = instantiate(cfg.data)
    model = instantiate(cfg.model)
    
    trainer = instantiate(cfg.trainer, model=model)
    
    train_loader, test_loader = data.get_dataloaders()
    
    trainer.train(
        train_loader=train_loader,
        test_loaders={"1":test_loader},
        optimizer=cfg.model.optimizer,
        scheduler=cfg.model.scheduler if 'scheduler' in cfg.model else None,
        regularizer=cfg.model.regularizer if 'regularizer' in cfg.model else None,
        training_loss=cfg.trainer.config.loss,
        eval_losses=[cfg.trainer.config.loss], # TODO: Support multiple eval losses
        save_every=cfg.trainer.config.save_every,
        save_dir=cfg.trainer.config.save_path,
        resume_from_dir=cfg.trainer.config.resume_from_dir if 'resume_from_dir' in cfg.trainer.config else None,
    )
    
    trainer.train()
    
if __name__ == "__main__":
    main()

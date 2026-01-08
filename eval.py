import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from evaluation.ns_evaluator import NSEvaluator

@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    """
    Main evaluation entry point.
    Initializes dataset, model, and evaluator, then runs evaluation.
    """
    # Initialize dataset
    data = instantiate(cfg.data)
    
    # Initialize model
    model = instantiate(cfg.model)
    
    # Create evaluator with full config
    # NSEvaluator handles model and data instantiation internally
    evaluator = NSEvaluator(cfg=cfg)
    
    # Run evaluation
    evaluator.run()
    
    return


if __name__ == "__main__":
    main()

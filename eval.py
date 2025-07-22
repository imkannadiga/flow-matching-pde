import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="eval", version_base=None)
def main(cfg: DictConfig):
    # TODO
    ## initialize dataset
    data = instantiate(cfg.data)
    ## initialize model
    model = instantiate(cfg.model)
    ## initialize the evaluation class with model, data and state_dict path
    eval_task = instantiate(cfg.eval)
    ## run
    eval_task.run()
    
    return
    
    
if __name__ == "__main__":
    main()

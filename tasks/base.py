import abc
from omegaconf import DictConfig
import hydra


class BaseTask(abc.ABC):
    def __init__(self, model, dataset, train: DictConfig, eval: DictConfig, wandb: DictConfig, **kwargs):
        """
        Base class for all tasks.
        Handles model and dataset instantiation via Hydra config.
        """
        self.train_cfg = train
        self.eval_cfg = eval
        
        print(f"Train Config: {train}")
        print(f"Eval Config: {eval}")

        # Instantiate model and dataset using Hydra
        self.model = model
        self.dataset = dataset
        self.wandb_cfg = wandb

        # Optionally log what we're running
        task_name = kwargs.get("name", self.__class__.__name__)
        print(f"[INFO] Initialized task: {task_name}")
        print(f"[INFO] Model: {self.model.__class__.__name__}")
        print(f"[INFO] Dataset: {self.dataset.__class__.__name__}")

    @abc.abstractmethod
    def run(self):
        """
        Must be implemented by child class.
        Should handle training, evaluation, logging, etc.
        """
        pass

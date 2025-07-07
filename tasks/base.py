import abc
from omegaconf import DictConfig
import hydra


class BaseTask(abc.ABC):
    def __init__(self, model: DictConfig, dataset: DictConfig, train: DictConfig, eval: DictConfig, **kwargs):
        """
        Base class for all tasks.
        Handles model and dataset instantiation via Hydra config.
        """
        self.train_cfg = train
        self.eval_cfg = eval

        # Instantiate model and dataset using Hydra
        self.model = hydra.utils.instantiate(model)
        self.dataset = hydra.utils.instantiate(dataset)

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

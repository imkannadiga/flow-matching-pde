from omegaconf import DictConfig
from hydra.utils import instantiate
from evaluation.eval_utils import load_model_from_manifest

class BaseEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = cfg.eval.device
        self.model = self._load_model().to(self.device).eval()
        self.dataset, self.loader_te = self._load_dataset()

    def _load_model(self):
        print("[Evaluator] Loading model from:", self.cfg.train.save_path)
        model = instantiate(self.cfg.model)
        return load_model_from_manifest(self.cfg.train.save_path, model_raw=model)

    def _load_dataset(self):
        raise NotImplementedError("Subclasses must implement _load_dataset")

    def run(self):
        raise NotImplementedError("Subclasses must implement run")

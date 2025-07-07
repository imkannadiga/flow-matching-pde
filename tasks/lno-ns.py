from tasks.base import BaseTask

class LNONSTask(BaseTask):
    def run(self):
        train_loader, test_loader = self.dataset.get_dataloaders()

        print(f"[INFO] Starting training on fno-ns with model {self.model.__class__.__name__}")
        # TODO: insert training logic here (e.g. self.train_model(train_loader, val_loader))
        print(f"[INFO] Training completed for fno-ns with model {self.model.__class__.__name__}")

        print(f"[INFO] Starting evaluation on fno-ns with model {self.model.__class__.__name__}")
        # TODO: insert evaluation logic here (e.g. self.evaluate_model(val_loader))
        print(f"[INFO] Evaluation completed for fno-ns with model {self.model.__class__.__name__}")
        
        return

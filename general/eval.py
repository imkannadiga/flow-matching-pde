import torch
from sklearn.metrics import mean_squared_error

def evaluate_model(model, dataloader):
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for x, y in dataloader:
            preds = model(x).squeeze().numpy()
            predictions.extend(preds)
            targets.extend(y.numpy())

    mse = mean_squared_error(targets, predictions)
    print(f"[Eval] MSE: {mse:.4f}")
    return mse
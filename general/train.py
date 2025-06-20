import torch.nn as nn
import torch.optim as optim
import wandb

def train_model(model: nn.Module, dataloader, cfg):
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    criterion = nn.MSELoss()
    
    metrics = {}

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            config=dict(cfg),  # Log full config as hyperparameters
        )
        wandb.watch(model, log="all")

    model.train()
    for epoch in range(cfg.train.epochs):
        total_loss = 0.0
        for x, y in dataloader:
            optimizer.zero_grad()
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        metrics[epoch+1]=avg_loss

        if cfg.wandb.enabled:
            wandb.log({"train-loss": avg_loss, "epoch": epoch + 1})

    if cfg.wandb.enabled:
            wandb.finish()

    return metrics

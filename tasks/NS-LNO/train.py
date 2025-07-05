import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from util.util import plot_samples, plot_loss_curve
import tqdm
import wandb
import time

def train_model(model_wr, cfg, train_loader, test_loader=None):
    optimizer = Adam(model_wr.model.parameters(), cfg.train.lr)
    scheduler = StepLR(optimizer, step_size=cfg.train.scheduler_step_size, gamma=cfg.train.scheduler_gamma)
    
    tr_losses = []
    te_losses = []
    eval_eps = []
    eval_int = cfg.train.eval_int
    save_int = cfg.train.save_int
    generate = cfg.train.generate
    evaluate = (eval_int > 0) and (test_loader is not None)

    model = model_wr.model
    device = model_wr.device

    first = True
    
    run = None

    if cfg.wandb.enabled:
        wandb.login()
        run = wandb.init(
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            project=cfg.wandb.project,
            group= cfg.wandb.model,
            config=dict(cfg),  # Log full config as hyperparameters
        )
        run.watch(model, log="all")

    epochs = cfg.train.epochs
    
    save_path = Path() / cfg.train.save_path
    checkpoint_path = Path() / cfg.train.save_path
    
    if(not save_path.exists()):
        save_path.mkdir(parents=True, exist_ok=True)
    if(not checkpoint_path.exists()):
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
    for ep in range(1, epochs+1):
            ##### TRAINING LOOP
            t0 = time.time()
            model.train()
            tr_loss = 0.0
            
            metrics = {}
            
            for batch, target in train_loader:
                batch = batch.to(device)
                target = target.to(device)
                batch_size = batch.shape[0]

                if first:
                    model_wr.n_channels = batch.shape[1]
                    model_wr.train_dims = batch.shape[2:]
                    first = False

                # Get model output
                model_out = model(batch)

                # Evaluate loss and do gradient step
                optimizer.zero_grad()
                loss = torch.mean( (model_out - target)**2 ) 
                loss.backward()
                optimizer.step()

                tr_loss += loss.item()

            tr_loss /= len(train_loader)
            tr_losses.append(tr_loss)
            if scheduler: scheduler.step()

            metrics["training_loss"] = tr_loss

            t1 = time.time()
            epoch_time = t1 - t0
            print(f'tr @ epoch {ep}/{epochs} | Loss {tr_loss:.6f} | {epoch_time:.2f} (s)')

            ##### EVAL LOOP
            if eval_int > 0 and (ep % eval_int == 0):
                t0 = time.time()
                eval_eps.append(ep)

                with torch.no_grad():
                    model.eval()

                    if evaluate:
                        te_loss = 0.0
                        for batch, target in test_loader:
                            batch = batch.to(device)
                            target = target.to(device)
                            batch_size = batch.shape[0]

                            x_pred = model(batch)

                            loss = torch.mean( (x_pred - target)**2 )

                            te_loss += loss.item()

                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')
                        
                        metrics['evaluation_loss'] = te_loss

            ##### BOOKKEEPING
            if ep % save_int == 0:
                torch.save(model.state_dict(), checkpoint_path / f'epoch_{ep}.pt')

            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')
            if cfg.wandb.enabled:
                run.log(metrics)
        
    if cfg.wandb.enabled:
        wandb.unwatch(model)
        run.finish()

    return
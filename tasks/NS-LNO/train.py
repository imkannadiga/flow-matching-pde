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

    if cfg.wandb.enabled:
        wandb.init(
            entity=cfg.wandb.entity if cfg.wandb.entity else None,
            project=cfg.wandb.project,
            config=dict(cfg),  # Log full config as hyperparameters
        )
        wandb.watch(model, log="all")

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
            
            for batch in train_loader:
                batch = batch.to(device)
                batch_size = batch.shape[0]

                if first:
                    model_wr.n_channels = batch.shape[1]
                    model_wr.train_dims = batch.shape[2:]
                    first = False

                # t ~ Unif[0, 1)
                t = torch.rand(batch_size, device=device)
                # Simluate p_t(x | x_1)
                x_noisy = model_wr.simulate(t, batch)
                # Get conditional vector fields
                target = model_wr.get_conditional_fields(t, batch, x_noisy)

                x_noisy = x_noisy.to(device)
                target = target.to(device)         

                # Get model output
                model_out = model(t, x_noisy)

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
                        for batch in test_loader:
                            batch = batch.to(device)
                            batch_size = batch.shape[0]

                            # t ~ Unif[0, 1)
                            t = torch.rand(batch_size, device=device)
                            # Simluate p_t(x | x_1)
                            x_noisy = model_wr.simulate(t, batch)
                            # Get conditional vector fields
                            target = model_wr.get_conditional_fields(t, batch, x_noisy)

                            x_noisy = x_noisy.to(device)
                            target = target.to(device)         
                            model_out = model(t, x_noisy)

                            loss = torch.mean( (model_out - target)**2 )

                            te_loss += loss.item()

                        te_loss /= len(test_loader)
                        te_losses.append(te_loss)

                        t1 = time.time()
                        epoch_time = t1 - t0
                        print(f'te @ epoch {ep}/{epochs} | Loss {te_loss:.6f} | {epoch_time:.2f} (s)')
                        
                        metrics['evaluation_loss'] = te_loss


                    # genereate samples during training?
                    if generate:
                        samples = model_wr.sample(model_wr.train_dims, n_channels=model_wr.n_channels, n_samples=16)
                        plot_samples(samples, save_path / f'samples_epoch{ep}.pdf')

            ##### BOOKKEEPING
            if ep % save_int == 0:
                torch.save(model.state_dict(), checkpoint_path / f'epoch_{ep}.pt')

            if evaluate:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf', te_loss=te_losses, te_epochs=eval_eps)
            else:
                plot_loss_curve(tr_losses, save_path / 'loss.pdf')
            
            wandb.log(metrics)
        
    wandb.finish()

    return
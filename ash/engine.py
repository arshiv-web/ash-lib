"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import time
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
from ash.utils import save_model, print_time
from datetime import datetime

# Try to import wandb (fail gracefully or strictly based on user preference)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch."""
    
    model.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device),y.to(device)
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc += accuracy_fn(y_true=y, y_pred=y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch."""
    
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device),y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            test_acc += accuracy_fn(y_true=y, y_pred=y_pred)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          accuracy_fn,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          early_stopper = None,
          project_name: str = "ash_experiment",
          run_name: str = "exp_01",
          extra: str = None,  # <--- NEW: Optional extra folder
          save_dir: str = "runs",
          log_interval: int = 1,
          disable_wandb: bool = False) -> Dict[str, List]:
    """
    Trains and tests a PyTorch model with W&B logging, checkpointing, and early stopping.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    run_dir = os.path.join(save_dir, timestamp, project_name, run_name)
    if extra:
        run_dir = os.path.join(run_dir, extra)
    
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Logging results to: {run_dir}")

    if not disable_wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is not installed! Run `pip install wandb` or pass `disable_wandb=True`.")
    
        wandb.init(project=project_name, name=run_name, config={
            "epochs": epochs,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "architecture": model.__class__.__name__,
            "date": timestamp,
            "tag": extra
        })
    else:
        print("[INFO] W&B explicitly disabled. Running in offline mode.")
    
    model.to(device)
    total_start_time = time.perf_counter()

    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_test_acc = 0.0

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if (epoch + 1) % log_interval == 0:
            print(
                f"\nEp: {epoch+1} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.2f}% | "
                f"Test: loss={test_loss:.4f} acc={test_acc:.2f}%"
            )
        
        if not disable_wandb and HAS_WANDB:
            wandb.log({
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
                "lr": optimizer.param_groups[0]['lr']
            })
        
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
        else:
            scheduler.step()
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model(model, run_dir, "best_model.pth")
        
        save_model(model, run_dir, "last_model.pth")

        if early_stopper:
            early_stopper(test_loss, model, path=run_dir)
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}")
                break
    
    total_end_time = time.perf_counter()
    print_time(total_start_time, total_end_time, device=str(device), context="Total Training")

    if not disable_wandb and HAS_WANDB:
        wandb.finish()

    return results

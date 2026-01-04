import os
import random
import numpy as np
import torch
from pathlib import Path
from torchinfo import summary

def seed_everything(seed: int = 42):
    """
    Locks the random seed for reproducibility across standard libraries.
    
    Args:
        seed (int): The seed number to use (default: 42).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[ash] Global seed set to {seed}")

def print_time(start: float, 
               end: float, 
               device: str = None, 
               context: str = "Computation") -> float:
    """
    Prints the difference between start and end time.
    
    Args:
        start (float): Start time (use time.perf_counter()).
        end (float): End time (use time.perf_counter()).
        device (str, optional): The device name (e.g., 'cuda', 'cpu').
        context (str, optional): Label for what is being timed (e.g. 'Train', 'Test').

    Returns:
        float: The total time in seconds.
    """
    total_time = end - start
    
    device_msg = f" on {device}" if device else ""
    print(f"\n[INFO] {context} time{device_msg}: {total_time:.3f} seconds")
    
    return total_time

def save_model(model: torch.nn.Module, 
               target_dir: str, 
               model_name: str):
    """Saves a PyTorch model to a target directory."""
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with .pt or .pth"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def print_model_summary(model: torch.nn.Module, input_size):
    summary(model=model, 
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
) 

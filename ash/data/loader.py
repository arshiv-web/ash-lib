import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Callable, List, Optional
from .dataset import AshFolderDataset

def seed_worker(worker_id):
    """
    Helper to ensure data loading is reproducible across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    loader: Callable,
    val_dir: str = None, 
    train_transform: Optional[Callable] = None, 
    test_transform: Optional[Callable] = None, 
    target_transform: Optional[Callable] = None,
    batch_size: int = 32, 
    num_workers: int = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    """
    Creates reproducible Train and Test DataLoaders from directories.
    
    Args:
        train_dir: Path to training data.
        test_dir: Path to testing data.
        val_dir: Path to validation data.
        loader: Function to load a single file (e.g. Image.open).
        train_transform: Transform pipeline (e.g. transforms.Compose).
        test_transform: Transform pipeline (e.g. transforms.Compose).
        target_transform: Target Transform pipeline (e.g. transforms.Compose).
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses. Defaults to os.cpu_count().
        seed: Random seed for reproducibility.

    Returns:
        train_dataloader, test_dataloader, val_dataloader, class_names
    """
    
    # 1. Hardware Setup
    if num_workers is None:
        num_workers = os.cpu_count()

    # 2. Generator Setup (Critical for Reproducibility)
    g = torch.Generator()
    g.manual_seed(seed)

    train_data = AshFolderDataset(root=train_dir, loader=loader, transform=train_transform, target_transform=target_transform)
    test_data = AshFolderDataset(root=test_dir, loader=loader, transform=test_transform, target_transform=target_transform)
    val_data = None
    if val_dir is not None:
        val_data = AshFolderDataset(root=val_dir, loader=loader, transform=test_transform, target_transform=target_transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_dataloader = None
    if val_dir is not None:
        val_dataloader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_dataloader, test_dataloader, val_dataloader, class_names
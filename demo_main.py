import ash
import torch

# 1. Setup
ash.seed_everything(42)

# 2. Get Data & Loaders
train_dl, test_dl, val_dl, classes = ash.create_dataloaders(
    train_dir="data/train",
    test_dir="data/test",
    batch_size=32
)

# 3. Train with one line
results = ash.train(
    model=model,
    train_dataloader=train_dl,
    test_dataloader=val_dl,
    epochs=10,
    project_name="My_Research_Project"
)
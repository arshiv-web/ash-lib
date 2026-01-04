import torch

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates accuracy between truth labels and predictions.
    
    Automatically handles:
      - Raw Logits: (Batch, Classes) -> takes argmax
      - Labels: (Batch) -> compares directly
    
    Args:
        y_true (torch.Tensor): Truth labels. Shape: [Batch]
        y_pred (torch.Tensor): Predictions (logits or labels). Shape: [Batch, Classes] or [Batch]

    Returns:
        float: Accuracy value (0.0 to 100.0)
    """
    
    # 1. Handle Device Mismatch (Move true labels to prediction device)
    # This prevents "Expected all tensors to be on the same device" errors
    y_true = y_true.to(y_pred.device)

    # 2. Handle Logits (Multi-class outputs)
    # If y_pred has an extra dimension (e.g., [32, 10]) compared to y_true ([32])
    # we assume they are logits and take the index of the highest score.
    if y_pred.ndim > 1 and y_true.ndim == 1:
        y_pred = torch.argmax(y_pred, dim=1)

    # 3. Verification
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    # 4. Calculation
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    
    return acc
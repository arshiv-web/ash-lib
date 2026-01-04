# ash-lib ⚡

A personal, high-performance PyTorch research library designed for reproducibility, experiment tracking, and rapid transfer learning.

## Features

* **Reproducible Data Loading**: "Seed Everything" logic baked into `DataLoaders` for deterministic training.
* **Professional Training Loop**: Integrated with **Weights & Biases (W&B)**, Automatic Mixed Precision (AMP), and Early Stopping.
* **Smart Checkpointing**: Automatically saves the `best_model.pth` (based on validation accuracy) and `last_model.pth`.
* **Custom Transforms**: Includes Letterbox Resizing to preserve image aspect ratios.
* **HPC Ready**: optimized for CUDA training loops.

## Installation

### Google Colab / Cloud
You can install `ash-lib` directly from GitHub in a single line:

```bash
!pip install git+[https://github.com/YOUR_USERNAME/ash-lib.git](https://github.com/YOUR_USERNAME/ash-lib.git)
# Task 1 — Architecture Swap (Beginner)

## Context

`demo_notebook.ipynb` trains a simple 2-layer CNN on the EuroSAT Sentinel-2 land-use dataset.
The `bonsai/` directory (cloned into Google Drive at `/content/drive/My Drive/demo_drive_folder/bonsai/`)
contains advanced vision architectures implemented in JAX.

## Your Task

Create a new notebook called `advanced_notebook.ipynb` that replaces the CNN with a more
powerful architecture from bonsai.

### Requirements

1. Browse `bonsai/` to identify at least two candidate architectures suitable for image
   classification. In a markdown cell, explain why you chose the one you did for a
   64×64 satellite imagery classification task with 10 classes.

2. Adapt the chosen architecture to:
   - Accept 64×64 RGB input (3 channels)
   - Output 10 logits (one per EuroSAT class)
   - Work within the existing training loop (same `train_step`, `eval_step` interface)

3. Keep all other pipeline code (data loading, optimizer, export) unchanged.

4. Confirm the model trains without errors and reaches higher test accuracy than the
   baseline CNN (~85% after 2000 steps).

### Constraints

- The Colab TPU kernel can only read files inside Google Drive.
  Import bonsai using a path relative to the Drive folder, not from pip.
- Do not change `batch_size`, `learning_rate`, or `train_steps` — keep comparisons fair.

## Deliverable

A working `advanced_notebook.ipynb` that runs end-to-end on TPU.

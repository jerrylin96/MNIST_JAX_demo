# Task 2 — Scientific Evaluation & Visualization (Intermediate)

## Context

The demo notebook logs only overall accuracy and loss. For earth science applications,
per-class performance matters — a model that perfectly classifies Forest but fails on
PermanentCrop vs. AnnualCrop is scientifically misleading.

## Your Task

Extend `demo_notebook.ipynb` (or your `advanced_notebook.ipynb`) with a thorough
evaluation section.

### Requirements

1. **Confusion matrix**: Compute and plot a normalized confusion matrix over the full
   test set. Label axes with the EuroSAT class names.

2. **Per-class metrics**: Report precision, recall, and F1-score for each of the 10 classes
   in a formatted table. Highlight the two most-confused class pairs.

3. **Error analysis**: Display a 5×5 grid of the highest-confidence misclassifications —
   show the true label, predicted label, and confidence score for each image.

4. **Spatial context note**: Add a markdown cell discussing why certain class pairs
   (e.g., AnnualCrop vs. PermanentCrop, River vs. SeaLake) are harder to distinguish
   from nadir-view RGB imagery, and what additional data (e.g., NIR band, temporal
   composites) could help.

### Hints

- `jax.nn.softmax` gives per-class probabilities from logits.
- `sklearn.metrics.confusion_matrix` and `classification_report` work on numpy arrays.
- Collect all predictions in a list during the eval loop, then concatenate.

## Deliverable

An updated notebook with a new "Evaluation" section containing all four items above.

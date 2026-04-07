# Task 3 — Scientifically Correct Data Pipeline (Intermediate)

## Context

The demo notebook splits EuroSAT randomly (80/20 by percentage). This is fine for a
benchmark but problematic for real geospatial workflows: satellite tiles from the same
geographic region are spatially autocorrelated, so a random split leaks information
between train and test.

## Your Task

Rebuild the data pipeline to be more scientifically rigorous.

### Requirements

1. **Class-balanced sampling**: EuroSAT is mildly imbalanced (Pasture has 2000 images,
   others up to 3000). Implement class-balanced batching so each training batch contains
   an equal number of examples from each class.

2. **Data augmentation**: Add domain-appropriate augmentations for satellite imagery:
   - Random horizontal and vertical flips (valid for nadir-view images)
   - Random 90° rotations (rotationally symmetric overhead imagery)
   - Color jitter ±10% (sensor calibration variation)
   Do NOT add augmentations that would be physically implausible (e.g., extreme distortion).

3. **Stratified split**: Replace the percentage-based split with a stratified split
   that ensures each class is proportionally represented in both train and test sets.

4. **Normalization**: Instead of dividing by 255, normalize each channel to zero mean
   and unit variance using the EuroSAT RGB channel statistics:
   - Mean: [0.3444, 0.3803, 0.4078]
   - Std:  [0.2026, 0.1366, 0.1148]

5. Add a markdown cell explaining why each of these choices matters for scientific
   validity vs. benchmark leaderboard performance.

### Constraints

- All augmentations must be implemented in `tf.data` (before `.batch()`) so they
  run on CPU and don't bottleneck the TPU.
- The test set must never see augmented images.

## Deliverable

An updated data loading section that passes a smoke test: the per-class sample counts
in each batch should be within ±2 of `batch_size / num_classes`.

# Task 4 — Uncertainty Quantification (Advanced)

## Context

Scientists rarely want just a class label — they need to know how confident the model is.
A land-cover map with calibrated uncertainty estimates is far more useful than one without,
because it tells downstream users where to trust automated predictions vs. where to request
human review or additional observations.

## Your Task

Add uncertainty quantification to the pipeline using two complementary methods.

### Requirements

1. **MC Dropout inference**: Enable dropout at inference time by calling `model.train()`
   instead of `model.eval()`. Run `N=50` stochastic forward passes for each test batch.
   - Compute the mean prediction (mean of softmax probabilities across passes)
   - Compute **predictive entropy** as the uncertainty metric:
     `H = -sum(p * log(p + 1e-8))` over classes
   - Report mean predictive entropy separately for correct vs. incorrect predictions.
     A well-calibrated model should have higher entropy on mistakes.

2. **Temperature scaling calibration**: After training, find the optimal temperature `T`
   that minimizes Expected Calibration Error (ECE) on the test set. Plot a reliability
   diagram (confidence vs. accuracy in 10 bins) before and after scaling.

3. **Uncertainty map**: Select 3 test images per class. For each, display:
   - The RGB image
   - The predicted class and confidence
   - A bar chart of the mean probability distribution across the 50 MC samples
   - The predictive entropy value

4. Add a markdown cell discussing how uncertainty estimates could be used to prioritize
   which satellite tiles to send for human annotation in an active learning workflow.

### Hints

- `jax.vmap` can vectorize the 50 forward passes efficiently on TPU.
- ECE can be computed with `sklearn.calibration.calibration_curve`.
- Temperature scaling: replace logits with `logits / T` and optimize `T` with optax.

## Deliverable

An `uncertainty_notebook.ipynb` with all four items above, runnable end-to-end on TPU.

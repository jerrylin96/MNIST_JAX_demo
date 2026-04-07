# Task 5 — Multispectral Sentinel-2 Input (Advanced)

## Context

The demo uses EuroSAT RGB (3 bands). However, Sentinel-2 provides 13 spectral bands
including near-infrared (NIR), shortwave infrared (SWIR), and red-edge bands that carry
information invisible to RGB cameras and are routinely used in vegetation indices (NDVI),
water indices (NDWI), and burn severity maps.

The `eurosat/all` TFDS variant provides all 13 bands as a (64, 64, 13) float32 tensor.

## Your Task

Extend the pipeline to exploit the full 13-band Sentinel-2 input.

### Requirements

1. **Switch to `eurosat/all`**: Update the data loading cell to use `eurosat/all`.
   Note that pixel values are now raw DN (digital numbers) in range ~0–10000, not
   normalized to [0,1]. Apply per-band normalization using the dataset statistics
   (compute mean and std from a 5000-sample subset before training).

2. **Adapt the model**: Update `conv1` to accept 13-channel input. If using a bonsai
   ViT, update the patch embedding layer's `in_channels`.

3. **Band importance analysis**: After training, implement a simple occlusion sensitivity
   experiment: zero out one band at a time across the entire test set and record the
   drop in accuracy. Plot a bar chart of accuracy drop per band, labeled with the
   Sentinel-2 band names (B01–B12, B8A). Which bands matter most for which classes?

4. **Spectral index features**: As a comparison, compute NDVI and NDWI from the
   appropriate bands and add them as two extra channels (making a 15-channel input).
   Report whether this improves accuracy.

5. Add a markdown cell interpreting the band importance results in terms of physical
   meaning (e.g., why NIR/SWIR bands matter for vegetation vs. urban discrimination).

### Sentinel-2 Band Reference

| Band | Wavelength | Resolution | Physical meaning |
|---|---|---|---|
| B02 | 490 nm | 10m | Blue |
| B03 | 560 nm | 10m | Green |
| B04 | 665 nm | 10m | Red |
| B08 | 842 nm | 10m | NIR |
| B05 | 705 nm | 20m | Red edge 1 |
| B06 | 740 nm | 20m | Red edge 2 |
| B07 | 783 nm | 20m | Red edge 3 |
| B8A | 865 nm | 20m | NIR narrow |
| B11 | 1610 nm | 20m | SWIR 1 |
| B12 | 2190 nm | 20m | SWIR 2 |
| B01 | 443 nm | 60m | Coastal aerosol |
| B09 | 945 nm | 60m | Water vapour |
| B10 | 1375 nm | 60m | Cirrus |

NDVI = (B08 - B04) / (B08 + B04 + 1e-6)
NDWI = (B03 - B08) / (B03 + B08 + 1e-6)

## Deliverable

A `multispectral_notebook.ipynb` with all five items, including a clear summary of which
spectral bands provide the most discriminative information for EuroSAT classification.

# Neutrino Direction Reconstruction

This repository is a cleaned portfolio version of a machine-learning workflow for neutrino arrival-direction reconstruction using simulated ANITA-III event data.

The code was developed around a research workflow that used timing-map inputs and signal-to-noise ratio features to predict incoming particle direction. The broader project is ongoing, and my access to the current project data, trained CNN model, and latest performance outputs ended when my role on the project ended. For that reason, this repository does **not** include raw data, model checkpoints, or current results.

This repo is intended as a technical code sample for job applications, not as a fully reproducible public research package.

---

## Project Overview

The goal of this workflow is to reconstruct the incoming direction of simulated neutrino events from detector-derived timing maps.

Each event includes:

- A two-channel timing map with shape `(2, 48, 150)`
- Signal-to-noise ratio features for vertical and horizontal polarization
- True angular labels:
  - `phi`: azimuthal direction
  - `theta`: elevation angle

The main model is a multi-input convolutional neural network that combines image-like timing-map information with SNR features. A Random Forest regressor is included as a simpler baseline model.

---

## Visual Examples

The figures below are representative visualizations from an earlier notebook prototype used to develop the workflow. They are included to show the structure of the data and the prediction task, not as current results from the ongoing project.

### Clean Timing Map Example

![Clean timing map example](<Figures/clean timing map.png>)

This example shows a cleaner timing-map pattern where signal structure is visible across the detector channels.

### Noise Timing Map Example

![Noise timing map example](<Figures/noise timing map.png>)

This example shows a noisier timing map where the signal is less visually distinct.

### Representative Direction Distribution

![Representative theta and phi distribution](<Figures/angle_distribution.png>)

The representative sample shows that `theta` is concentrated near the horizon, especially between roughly `0` and `-10` degrees, then trails off at lower elevations. In this geometry, exactly `0` degrees corresponds to no ice path, while events just below the horizon are more relevant for neutrino detection. The `phi` distribution is more evenly spread across azimuth.

---

## Modeling Approach

### Label Encoding

Instead of predicting `phi` directly as a single angle, the workflow encodes azimuth using sine and cosine components:

```python
labels[:, 0] = cos(phi)
labels[:, 1] = sin(phi)
labels[:, 2] = theta / 90
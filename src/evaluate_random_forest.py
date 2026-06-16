"""
Evaluate the Random Forest regressor for neutrino arrival-direction reconstruction.

The script loads the trained Random Forest model, predicts test-set directions, calculates angular error, and saves error data plus a histogram.
"""

# ----------------------------- LIBRARIES -----------------------------

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- CONFIG -----------------------------

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

TEST_FEATURES_PATH = f"{DATA_DIR}/test_features_rf.npy"
TEST_LABELS_PATH = f"{DATA_DIR}/test_labels_rf.npy"

MODEL_PATH = f"{MODEL_DIR}/random_forest.joblib"

ERROR_OUTPUT_PATH = f"{OUTPUT_DIR}/random_forest_errors.npz"
FIGURE_OUTPUT_PATH = f"{FIGURE_DIR}/random_forest_angular_error.pdf"

# ----------------------------- PERFORMANCE FUNCTIONS -----------------------------

def labelsToAngles(labels):

    # Convert phi to degrees
    phi = np.rad2deg(np.arctan2(labels[:, 1], labels[:, 0]))
    phi = phi % 360

    # Convert theta to degrees
    theta = labels[:, 2] * 90.0

    return np.column_stack([phi, theta])


def calcAngularError(trueLabels, predLabels):

    # Convert labels into angles
    trueAngles = labelsToAngles(trueLabels)
    predAngles = labelsToAngles(predLabels)

    # Compute phi error
    dphi = (predAngles[:, 0] - trueAngles[:, 0] + 180) % 360 - 180

    # Compute theta error
    dtheta = predAngles[:, 1] - trueAngles[:, 1]

    # Approximate total angular error
    angularError = np.sqrt(dphi**2 + dtheta**2)

    return trueAngles, predAngles, dphi, dtheta, angularError


def saveErrorFigure(angularError):

    plt.figure(figsize=(7, 5))
    plt.hist(angularError, bins=60)
    plt.xlabel("Angular error [deg]")
    plt.ylabel("Number of events")
    plt.title("Random Forest angular error")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_PATH)
    plt.close()

# ----------------------------- RUN SCRIPT -----------------------------

if __name__ == "__main__":

    # Make output folders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Load test data/trained model
    testFeatures = np.load(TEST_FEATURES_PATH)
    testLabels = np.load(TEST_LABELS_PATH)
    rfEstimator = joblib.load(MODEL_PATH)

    print(f"RF test features: {testFeatures.shape}")
    print(f"RF test labels: {testLabels.shape}\n")

    # Predict labels
    predLabels = rfEstimator.predict(testFeatures)

    # Calculate angular errors
    trueAngles, predAngles, dphi, dtheta, angularError = calcAngularError(
        testLabels,
        predLabels,
    )

    print("Random Forest results:")
    print(f"Number of test events: {len(angularError)}")
    print(f"Mean angular error: {np.mean(angularError):.3f} deg")
    print(f"Std angular error: {np.std(angularError):.3f} deg")
    print(f"Median angular error: {np.median(angularError):.3f} deg\n")

    # Save error data
    np.savez(
        ERROR_OUTPUT_PATH,
        trueAngles=trueAngles,
        predAngles=predAngles,
        dphi=dphi,
        dtheta=dtheta,
        angularError=angularError,
    )

    # Save histogram
    saveErrorFigure(angularError)

    print(f"Saved error data to: {ERROR_OUTPUT_PATH}")
    print(f"Saved figure to: {FIGURE_OUTPUT_PATH}")
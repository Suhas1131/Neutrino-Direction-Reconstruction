"""
Evaluate the CNN for neutrino arrival-direction reconstruction.

The script loads the trained CNN model, predicts test-set directions, calculates angular error, and saves error data plus a histogram.
"""

# ----------------------------- LIBRARIES -----------------------------

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import Sequence

# ----------------------------- PATHS -----------------------------

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
FIGURE_DIR = "figures"

TEST_TIMING_MAP_PATH = f"{DATA_DIR}/test_timingMap_cnn.npy"
TEST_SNR_PATH = f"{DATA_DIR}/test_snr_cnn.npy"
TEST_LABELS_PATH = f"{DATA_DIR}/test_labels_cnn.npy"

MODEL_PATH = f"{MODEL_DIR}/anita_CNN.keras"

ERROR_OUTPUT_PATH = f"{OUTPUT_DIR}/cnn_errors.npz"
FIGURE_OUTPUT_PATH = f"{FIGURE_DIR}/cnn_angular_error.pdf"

BATCH_SIZE = 512

# ----------------------------- DATA GENERATOR -----------------------------

class AnitaEvalGenerator(Sequence):

    def __init__(self, timingMap, snr, batchSize):
        self.timingMap = timingMap
        self.snr = snr
        self.batchSize = batchSize

    def __len__(self):
        return int(np.ceil(len(self.snr) / self.batchSize))

    def __getitem__(self, idx):
        # Select batch
        start = idx * self.batchSize
        stop = min((idx + 1) * self.batchSize, len(self.snr))

        # Convert timing maps
        inputMaps = self.timingMap[start:stop]
        inputMaps = inputMaps.transpose(0, 2, 3, 1)

        # Load SNR features
        inputSnr = self.snr[start:stop]

        return (inputMaps, inputSnr)

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

    plt.figure(figsize=(8, 6))
    plt.hist(angularError, bins=100)
    plt.xlabel("Angular error [deg]")
    plt.ylabel("Number of events")
    plt.title("CNN angular error", weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURE_OUTPUT_PATH)
    plt.close()

# ----------------------------- RUN SCRIPT -----------------------------

if __name__ == "__main__":

    # Make output folders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    # Load test data
    testTimingMap = np.load(TEST_TIMING_MAP_PATH, mmap_mode="r")
    testSnr = np.load(TEST_SNR_PATH, mmap_mode="r")
    testLabels = np.load(TEST_LABELS_PATH, mmap_mode="r")

    print(f"CNN test timing maps: {testTimingMap.shape}")
    print(f"CNN test SNR: {testSnr.shape}")
    print(f"CNN test labels: {testLabels.shape}\n")

    # Load trained CNN
    model = tf.keras.models.load_model(MODEL_PATH)

    testGen = AnitaEvalGenerator(
        testTimingMap,
        testSnr,
        batchSize=BATCH_SIZE,
    )

    phiPred, thetaPred = model.predict(testGen, verbose=1)

    # Combine CNN outputs
    predLabels = np.concatenate([phiPred, thetaPred], axis=1)

    # Calculate angular errors
    trueAngles, predAngles, dphi, dtheta, angularError = calcAngularError(
        testLabels,
        predLabels,
    )

    print("CNN results:")
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
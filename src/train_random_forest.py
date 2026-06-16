"""
Train a Random Forest regressor for neutrino arrival-direction reconstruction.

The model uses flattened timing-map features and SNR values to predict encoded direction labels.

Input files:
    data/train_features_rf.npy -> flattened timing maps + SNR
    data/train_labels_rf.npy -> [cos(phi), sin(phi), theta / 90]
"""

# ----------------------------- LIBRARIES -----------------------------

import os
import time
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor

# ----------------------------- CONFIG -----------------------------

DATA_DIR = "data"
MODEL_DIR = "models"

TRAIN_FEATURES_PATH = f"{DATA_DIR}/train_features_rf.npy"
TRAIN_LABELS_PATH = f"{DATA_DIR}/train_labels_rf.npy"

MODEL_OUTPUT_PATH = f"{MODEL_DIR}/random_forest.joblib"

N_ESTIMATORS = 100
RANDOM_SEED = 42
N_JOBS = -1

# ----------------------------- TRAINING -----------------------------

if __name__ == "__main__":

    # Make directory to store model
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load preprocessed Random Forest training arrays
    trainFeatures = np.load(TRAIN_FEATURES_PATH)
    trainLabels = np.load(TRAIN_LABELS_PATH)

    print(f"train features: {trainFeatures.shape}")
    print(f"train labels: {trainLabels.shape}\n")

    # Define Random Forest regressor
    rfEstimator = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1,
    )

    startTime = time.time()

    # Train model
    rfEstimator.fit(trainFeatures, trainLabels)

    print(f"Training time: {time.time() - startTime:.3f} seconds\n")

    # Save trained model
    joblib.dump(rfEstimator, MODEL_OUTPUT_PATH)
    print(f"Saved random forest to: {MODEL_OUTPUT_PATH}")
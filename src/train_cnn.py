"""
Train a CNN for neutrino arrival-direction reconstruction.

The model uses timing-map inputs and SNR features to predict direction labels.

The original workflow was designed for GPU training on HPC systems.

Input files:
    data/train_timingMap_cnn.npy   shape: (N, 2, 48, 150)
    data/train_snr_cnn.npy         shape: (N, 2)
    data/train_labels_cnn.npy      shape: (N, 3)

Labels:
    labels[:, 0] = cos(phi)
    labels[:, 1] = sin(phi)
    labels[:, 2] = scaled theta
"""

# Import Libraries
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence

# Directories
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"

# Paths to input files/labels
TRAIN_TIMING_MAP_PATH = os.path.join(DATA_DIR, "train_timingMap_cnn.npy")
TRAIN_SNR_PATH = os.path.join(DATA_DIR, "train_snr_cnn.npy")
TRAIN_LABELS_PATH = os.path.join(DATA_DIR, "train_labels_cnn.npy")

# Path to trained model
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "anita_CNN.keras")

BATCH_SIZE = 512
EPOCHS = 100
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.2
RANDOM_SEED = 42

# Set to an integer for debugging
N_DEBUG = None

# ----------------------------- DATA GENERATOR -----------------------------

class AnitaGenerator(Sequence):
    
    def __init__(self, timingMap, snr, labels, indices, batchSize, shuffle=True):
        self.timingMap = timingMap
        self.snr = snr
        self.labels = labels
        self.indices = np.array(indices)
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        # Number of batches per epoch
        return int(np.ceil(len(self.indices) / self.batchSize))

    def __getitem__(self, idx):
        # Select event indices for one batch
        batchIds = self.indices[idx * self.batchSize:(idx + 1) * self.batchSize]

        # Load timing maps and convert to (batch, height, width, channels)
        inputMaps = self.timingMap[batchIds]
        inputMaps = inputMaps.transpose(0, 2, 3, 1)

        # Load SNR features and labels
        inputSnr = self.snr[batchIds]
        label = self.labels[batchIds]

        # Split labels
        phi = label[:, :2]
        theta = label[:, 2:3]

        return (inputMaps, inputSnr), {"phi": phi, "theta": theta}

    def on_epoch_end(self):
        # Shuffle training indices after each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

# ----------------------------- MODEL -----------------------------

def convBlock(x, kernels, dropoutRate=0.15):
    
    x = Conv2D(kernels, 3, padding="same",
        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(kernels, 3, padding="same",
        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = MaxPooling2D()(x)  # downsample feature maps 
    x = Dropout(dropoutRate)(x)  # reduce overfitting

    return x

def buildModel():
    
    timingInput = Input(shape=(48, 150, 2), name="timing_map")

    x = convBlock(timingInput, 32, dropoutRate=0.10)
    x = convBlock(x, 64, dropoutRate=0.15)
    x = convBlock(x, 128, dropoutRate=0.20)

    x = Conv2D(128, 3, padding="same",
        kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)

    snrInput = Input(shape=(2,), name="snr")

    y = Dense(
        32, kernel_regularizer=l2(1e-4)
    )(snrInput)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    xy = Concatenate()([x, y])

    xy = Dense(
        128, kernel_regularizer=l2(1e-4)
    )(xy)
    xy = BatchNormalization()(xy)
    xy = Activation("relu")(xy)
    xy = Dropout(0.3)(xy)

    xy = Dense(
        64, kernel_regularizer=l2(1e-4)
    )(xy)
    xy = BatchNormalization()(xy)
    xy = Activation("relu")(xy)

    phiOutput = Dense(2, activation="linear", name="phi")(xy)
    thetaOutput = Dense(1, activation="linear", name="theta")(xy)

    model = Model(
        inputs=[timingInput, snrInput],
        outputs=[phiOutput, thetaOutput],
    )

    model.compile(
        optimizer=Adam(LEARNING_RATE),
        loss={
            "phi": "mse",
            "theta": "mse",
        },
        loss_weights={
            "phi": 0.7,
            "theta": 0.3,
        },
    )

    return model


# ----------------------------- TRAINING -----------------------------

def loadMemmap():

    # Open memory maps 
    trainTimingMap = np.lib.format.open_memmap(TRAIN_TIMING_MAP_PATH, mode="r")
    trainSnr = np.lib.format.open_memmap(TRAIN_SNR_PATH, mode="r")
    trainLabels = np.lib.format.open_memmap(TRAIN_LABELS_PATH, mode="r")

    # Print feature/label shapes
    print(f"Timing-map shape: {trainTimingMap.shape}")
    print(f"SNR shape: {trainSnr.shape}")
    print(f"Label shape: {trainLabels.shape}\n")

    return trainTimingMap, trainSnr, trainLabels


def trainValSplit(nEvents):

    indices = np.arange(nEvents)

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(indices)

    # Limit dataset for debugging
    if N_DEBUG is not None:
        indices = indices[:N_DEBUG]

    # Split train/validation datasets
    nTrain = int((1 - VALIDATION_FRACTION) * len(indices))

    trainIdx = indices[:nTrain]
    valIdx = indices[nTrain:]

    print(f"Training events: {len(trainIdx)}")
    print(f"Validation events: {len(valIdx)}\n")

    return trainIdx, valIdx

# ----------------------------- RUNNING SCRIPT -----------------------------

if __name__ == "__main__":

    # Make directories to store model/logs
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Open features/labels
    trainTimingMap, trainSnr, trainLabels = loadMemmap()

    nEvents = trainLabels.shape[0]
    trainIdx, valIdx = trainValSplit(nEvents)

    # Load one chunk at a time for training
    trainGen = AnitaGenerator(
        trainTimingMap,
        trainSnr,
        trainLabels,
        trainIdx,
        batchSize=BATCH_SIZE,
        shuffle=True,
    )

    # Load one chunk at a time for validation
    valGen = AnitaGenerator(
        trainTimingMap,
        trainSnr,
        trainLabels,
        valIdx,
        batchSize=BATCH_SIZE,
        shuffle=False,
    )

    model = buildModel()
    model.summary()  # Print model summary

    callbacks = [
        # stop if loss becomes invalid
        tf.keras.callbacks.TerminateOnNaN(),

        # save the best model
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUTPUT_PATH,
            save_best_only=True,
            monitor="val_loss",
            verbose=1,
        ),

        # lower learning rate if val_loss plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),

        # stop if val_loss does not improve
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor="val_loss",
            verbose=1,
        ),
    ]

    # Begin Training
    model.fit(
        trainGen,
        validation_data=valGen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_OUTPUT_PATH)
    print(f"Saved model to: {MODEL_OUTPUT_PATH}")
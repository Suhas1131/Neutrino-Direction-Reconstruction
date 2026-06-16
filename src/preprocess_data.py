
"""
Preprocess simulated ANITA-III event data for Random Forest and CNN models.

Inputs are pickle files containing timing maps, SNR values, and true arrival directions. Outputs are NumPy arrays used by the training scripts.

Expected dataframe columns:
    timingMap: array with shape (2, 48, 150)
    snr_vpol: vertical-polarization SNR
    snr_hpol: horizontal-polarization SNR
    phi_true: true phi angle in degrees
    theta_true: true theta angle in degrees
"""

# ----------------------------- LIBRARIES -----------------------------

import gc
import numpy as np
import pandas as pd

# ----------------------------- CONFIG -----------------------------

# paths to data directories
RAW_DATA_DIR = "/path/to/anita/pickle/files"
OUTPUT_DIR = "data"

RF_FILE = "anita_mc_summary_2k.pkl"

CNN_TRAIN_FILES = [
    "anita_mc_summary_large_0.pkl",
    "anita_mc_summary_large_1.pkl",
    "anita_mc_summary_large_2.pkl",
]

CNN_TEST_FILES = ["anita_mc_summary_large_3.pkl"]

TIMING_MAP_SHAPE = (2, 48, 150)
RF_TEST_FRACTION = 0.2
RANDOM_SEED = 42
CHUNK_SIZE = 2000

# ----------------------------- PREPROCESSING FUNCTIONS -----------------------------

def countRows(path):

    df = pd.read_pickle(path)
    nRows = len(df)

    # Clear memory
    del df
    gc.collect()
    
    return nRows

def normalizeTimingMaps(timingMap):

    # make signals stand out from noise
    timingMapSq = np.square(timingMap, dtype=np.float32)

    maxIntensity = np.max(timingMapSq, axis=(1, 2, 3), keepdims=True)
    maxIntensity[maxIntensity == 0] = 1.0  # Prevent division by 0

    timingMapSq = timingMapSq / maxIntensity

    return timingMapSq.astype(np.float32)

def normalizeLabels(df):
    
    phi = np.deg2rad(df["phi_true"].to_numpy(np.float32))  # Heading
    theta = df["theta_true"].to_numpy(np.float32)  # Elevation

    labels = np.empty((len(df), 3), dtype=np.float32)
    labels[:, 0] = np.cos(phi)
    labels[:, 1] = np.sin(phi)
    labels[:, 2] = theta / 90.0

    return labels

def normalizeSnr(df):
    
    snr = df[["snr_vpol", "snr_hpol"]].to_numpy(np.float32)
    snr = np.log10(snr)

    return snr.astype(np.float32)

# ----------------------------- RANDOM FOREST DATA -----------------------------

def splitDataRf(pklPath, testFraction=0.2, seed=42):
    
    df = pd.read_pickle(pklPath)
    nEvents = len(df)

    # Flatten timing maps
    timingMap = np.stack(df["timingMap"].to_numpy(), axis=0).astype(np.float32)
    timingMapSq = normalizeTimingMaps(timingMap)
    timingMapFlat = timingMapSq.reshape(nEvents, -1)

    snr = normalizeSnr(df)
    labels = normalizeLabels(df)

    features = np.concatenate([timingMapFlat, snr], axis=1).astype(np.float32)

    np.random.seed(seed)
    indices = np.arange(nEvents)
    np.random.shuffle(indices)

    nTest = int(np.round(testFraction * nEvents))
    testIdx = indices[:nTest]
    trainIdx = indices[nTest:]

    trainFeatures = features[trainIdx]
    trainLabels = labels[trainIdx]
    testFeatures = features[testIdx]
    testLabels = labels[testIdx]

    # Clear memory
    del df, timingMap, timingMapSq, timingMapFlat, snr, labels, features
    gc.collect()

    return trainFeatures, trainLabels, testFeatures, testLabels

def writeRfData():
    
    rfPath = f"{RAW_DATA_DIR}/{RF_FILE}"

    trainFeatures, trainLabels, testFeatures, testLabels = splitDataRf(
        rfPath,
        testFraction=RF_TEST_FRACTION,
        seed=RANDOM_SEED
    )

    np.save(f"{OUTPUT_DIR}/train_features_rf.npy", trainFeatures)
    np.save(f"{OUTPUT_DIR}/train_labels_rf.npy", trainLabels)
    np.save(f"{OUTPUT_DIR}/test_features_rf.npy", testFeatures)
    np.save(f"{OUTPUT_DIR}/test_labels_rf.npy", testLabels)

    print("Saved Random Forest arrays.")
    print(f"RF train features: {trainFeatures.shape}")
    print(f"RF train labels: {trainLabels.shape}")
    print(f"RF test features: {testFeatures.shape}")
    print(f"RF test labels: {testLabels.shape}\n")

# ----------------------------- CNN DATA -----------------------------

def saveCnnData(pklPaths, mapsArr, snrArr, labelsArr, offset0=0, chunkSize=2000):
    
    offset = offset0

    for path in pklPaths:
        print(f"Loading {path}...")

        df = pd.read_pickle(path)
        nRows = len(df)

        # Process the file in chunks
        for start in range(0, nRows, chunkSize):
            stop = min(start + chunkSize, nRows)
            nChunk = stop - start

            timingMap = df["timingMap"][start:stop].to_numpy()
            timingMap = np.stack(timingMap, axis=0).astype(np.float32)
            timingMapSq = normalizeTimingMaps(timingMap)

            dfChunk = df[start:stop]
            snr = normalizeSnr(dfChunk)
            labels = normalizeLabels(dfChunk)

            mapsArr[offset:offset + nChunk] = timingMapSq
            snrArr[offset:offset + nChunk] = snr
            labelsArr[offset:offset + nChunk] = labels

            offset += nChunk

            del timingMap, timingMapSq, snr, labels, dfChunk
            gc.collect()

        del df
        gc.collect()

    # Save memory-maps to disk
    mapsArr.flush()
    snrArr.flush()
    labelsArr.flush()

    return offset


def writeCnnData():
    
    trainPaths = [f"{RAW_DATA_DIR}/{fileName}" for fileName in CNN_TRAIN_FILES]
    testPaths = [f"{RAW_DATA_DIR}/{fileName}" for fileName in CNN_TEST_FILES]

    nTrain = sum(countRows(path) for path in trainPaths)
    nTest = sum(countRows(path) for path in testPaths)

    print(f"Number of CNN training events: {nTrain}")
    print(f"Number of CNN testing events:  {nTest}\n")

    # Open memory-maps
    trainTimingMap = np.lib.format.open_memmap(
        f"{OUTPUT_DIR}/train_timingMap_cnn.npy",
        mode="w+",
        dtype="float32",
        shape=(nTrain, *TIMING_MAP_SHAPE),
    )

    trainSnr = np.lib.format.open_memmap(
        f"{OUTPUT_DIR}/train_snr_cnn.npy",
        mode="w+",
        dtype="float32",
        shape=(nTrain, 2),
    )

    trainLabels = np.lib.format.open_memmap(
        f"{OUTPUT_DIR}/train_labels_cnn.npy",
        mode="w+",
        dtype="float32",
        shape=(nTrain, 3),
    )

    testTimingMap = np.lib.format.open_memmap(
        f"{OUTPUT_DIR}/test_timingMap_cnn.npy",
        mode="w+",
        dtype="float32",
        shape=(nTest, *TIMING_MAP_SHAPE),
    )

    testSnr = np.lib.format.open_memmap(
        f"{OUTPUT_DIR}/test_snr_cnn.npy",
        mode="w+",
        dtype="float32",
        shape=(nTest, 2),
    )

    testLabels = np.lib.format.open_memmap(
        f"{OUTPUT_DIR}/test_labels_cnn.npy",
        mode="w+",
        dtype="float32",
        shape=(nTest, 3),
    )

    print("Processing CNN training data.")
    trainWritten = saveCnnData(
        trainPaths,
        trainTimingMap,
        trainSnr,
        trainLabels,
        offset0=0,
        chunkSize=CHUNK_SIZE,
    )
    print("Memory map created: CNN training data.\n")

    print("Processing CNN testing data.")
    testWritten = saveCnnData(
        testPaths,
        testTimingMap,
        testSnr,
        testLabels,
        offset0=0,
        chunkSize=CHUNK_SIZE,
    )
    print("Memory map created: CNN testing data.\n")

    print(f"Training events written: {trainWritten}, expected: {nTrain}")
    print(f"Testing events written:  {testWritten}, expected: {nTest}")

# ----------------------------- RUN SCRIPT -----------------------------

if __name__ == "__main__":
    
    writeRfData()
    writeCnnData()
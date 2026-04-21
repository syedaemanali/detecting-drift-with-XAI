import io
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def download_dataset():
    "Download and extract the UCI Bank Marketing CSV if not already present."
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_DATA_PATH.exists():
        log.info("Dataset already on disk, skipping download")
        return

    log.info("Downloading UCI Bank Marketing dataset")
    response = requests.get(config.UCI_DOWNLOAD_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        target_file = "bank-additional/bank-additional-full.csv"
        with zf.open(target_file) as src, open(config.RAW_DATA_PATH, "wb") as dst:
            dst.write(src.read())

    log.info("Dataset saved to %s", config.RAW_DATA_PATH)


def load_and_preprocess():
    "Full preprocessing pipeline: encode, scale, SMOTE. Returns train/test splits."
    if config.PROCESSED_PATH.exists():
        log.info("Loading cached processed data")
        df = pd.read_parquet(config.PROCESSED_PATH)
        return _split_dataframe(df, apply_smote=True)

    download_dataset()
    df = pd.read_csv(config.RAW_DATA_PATH, sep=";")
    log.info("Raw data loaded, shape %s", df.shape)

    df = df[config.FEATURE_COLS + [config.TARGET_COL]].copy()
    df[config.TARGET_COL] = (df[config.TARGET_COL] == "yes").astype(int)

    # Encode all object columns using label encoding
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    numeric_cols = [c for c in config.FEATURE_COLS if c not in categorical_cols]
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PATH, index=False)
    log.info("Processed data cached at %s", config.PROCESSED_PATH)

    return _split_dataframe(df, apply_smote=True)


def load_data_variants():
    """Return both raw and SMOTE-balanced training splits from the same preprocessed dataframe."""
    if config.PROCESSED_PATH.exists():
        log.info("Loading cached processed data")
        df = pd.read_parquet(config.PROCESSED_PATH)
    else:
        download_dataset()
        df = pd.read_csv(config.RAW_DATA_PATH, sep=";")
        log.info("Raw data loaded, shape %s", df.shape)

        df = df[config.FEATURE_COLS + [config.TARGET_COL]].copy()
        df[config.TARGET_COL] = (df[config.TARGET_COL] == "yes").astype(int)

        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col].astype(str))

        numeric_cols = [c for c in config.FEATURE_COLS if c not in categorical_cols]
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

        config.PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.PROCESSED_PATH, index=False)
        log.info("Processed data cached at %s", config.PROCESSED_PATH)

    raw_split = _split_dataframe(df, apply_smote=False)
    smote_split = _split_dataframe(df, apply_smote=True)

    return {"raw": raw_split, "smote": smote_split}


def _split_dataframe(df, apply_smote=True):
    X = df[config.FEATURE_COLS].values
    y = df[config.TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    if apply_smote:
        # Class imbalance (~11% positive), SMOTE balances training set only
        smote = SMOTE(random_state=config.RANDOM_STATE)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        log.info(
            "Train size %d (after SMOTE), test size %d, positive rate %.2f%%",
            len(X_train),
            len(X_test),
            y_test.mean() * 100,
        )
    else:
        log.info(
            "Train size %d (raw), test size %d, positive rate %.2f%%",
            len(X_train),
            len(X_test),
            y_test.mean() * 100,
        )

    return X_train, X_test, y_train, y_test


def get_feature_names():
    return config.FEATURE_COLS
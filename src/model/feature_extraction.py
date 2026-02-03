import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from tokenizer.tokenizer_whitespace import TokenizerWhitespace
from vectorizer.vectorizer_bucket import VectorizerBucket
from vectorizer.vectorizer_label import VectorizerLabel
from vectorizer.vectorizer_sequential import VectorizerSequential

logger = logging.getLogger("experiment")


FEATURES = {
    "Title": Pipeline(
        [
            ("tokenizer", TokenizerWhitespace()),
            ("vectorizer", VectorizerSequential(8500, 20)),
        ],
    ),
    "Description": Pipeline(
        [
            ("tokenizer", TokenizerWhitespace()),
            ("vectorizer", VectorizerSequential(5000, 30)),
        ],
    ),
    "Categories": Pipeline(
        [
            ("vectorizer", VectorizerLabel()),
        ],
    ),
    "Duration": Pipeline(
        [
            ("vectorizer", VectorizerBucket([0, 60, 180, 360, 9999999])),
        ],
    ),
}


def transform_features(df, fit=False) -> dict:
    outputs = {}
    for col_name, pipeline in FEATURES.items():
        if fit:
            outputs[col_name] = pipeline.fit_transform(df[col_name]).to_numpy()
        else:
            outputs[col_name] = pipeline.transform(df[col_name]).to_numpy()

    passthroug_cols = [c for c in df.columns if c not in outputs]
    for col_name in passthroug_cols:
        outputs[col_name] = df[col_name].to_numpy()

    return outputs


def save_split(path, X, y):
    np.savez_compressed(
        path,
        YT_ID=X["YT ID"],
        Title=X["Title"],
        Description=X["Description"],
        Categories=X["Categories"],
        Duration=X["Duration"],
        Labels=y,
    )


def prepare_datasets_and_pipelines() -> None:
    train_df = pd.read_json("dataset/p3_dataset_split.json")
    val_df = pd.read_json("dataset/p3_dataset_validation.json")
    test_df = pd.read_json("dataset/p3_dataset_test.json")

    X_train = transform_features(train_df, fit=True)
    X_val = transform_features(val_df, fit=False)
    X_test = transform_features(test_df, fit=False)

    save_split("dataset/p4_train.npz", X_train, train_df["Is Music"].to_numpy())
    save_split("dataset/p4_val.npz", X_val, val_df["Is Music"].to_numpy())
    save_split("dataset/p4_test.npz", X_test, test_df["Is Music"].to_numpy())

    for name, pipeline in FEATURES.items():
        joblib.dump(pipeline, f"model/pipelines/{name}_pipeline.joblib")

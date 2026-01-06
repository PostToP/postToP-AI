import logging

import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer

from data.text_cleaning import DatasetPreprocessor, generate_new_pipeline

load_dotenv()

logger = logging.getLogger("experiment")


def preprocess_dataset() -> pd.DataFrame:
    dataset = pd.read_json("dataset/videos.json")

    dataset = dataset.dropna(subset=["Is Music"])

    # Initial data cleaning
    dataset["Title"] = dataset["Title"].fillna("")
    dataset["Description"] = dataset["Description"].fillna("")

    # Process text columns
    column_names = dataset.columns.tolist()
    text_pipeline = generate_new_pipeline()
    transformer = ColumnTransformer(
        [
            ("yt_id", "passthrough", ["YT ID"]),
            ("title", text_pipeline, "Title"),
            ("description", text_pipeline, "Description"),
        ],
        remainder="passthrough",
        verbose=True,
    )
    processed_dataset = transformer.fit_transform(dataset)
    processed_dataset = pd.DataFrame(processed_dataset, columns=column_names)
    len_before = len(processed_dataset)
    processed_dataset = DatasetPreprocessor.remove_similar_rows(processed_dataset)
    len_after = len(processed_dataset)
    logger.info(f"Removed {len_before - len_after} similar rows.")

    processed_dataset.to_json("dataset/p2_dataset.json", index=False)

    return processed_dataset

import pandas as pd
from sklearn.compose import ColumnTransformer

from data.text_cleaning import DatasetPreprocessor, generate_new_pipeline


def preprocess_dataset() -> pd.DataFrame:
    dataset = pd.read_json("dataset/videos.json")

    dataset = dataset.dropna(subset=["Is Music"])

    # Initial data cleaning
    dataset["Title"] = dataset["Title"].fillna("")
    dataset["Description"] = dataset["Description"].fillna("")

    # Process text columns
    column_names = dataset.columns.tolist()
    text_pipeline = generate_new_pipeline()
    tranformer = ColumnTransformer(
        [
            ("yt_id", "passthrough", ["YT ID"]),
            ("title", text_pipeline, "Title"),
            ("description", text_pipeline, "Description"),
        ],
        remainder="passthrough",
    )
    train_df = tranformer.fit_transform(dataset)
    train_df = pd.DataFrame(train_df, columns=column_names)
    train_df = DatasetPreprocessor.remove_similar_rows(train_df)

    train_df.to_json("dataset/p2_dataset.json", index=False)

    return train_df

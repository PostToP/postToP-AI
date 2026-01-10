import pandas as pd

from data.text_cleaning import split_dataset


def split_dataset() -> None:
    df = pd.read_json("dataset/p2_dataset.json")
    train_df, val_df = split_dataset(df, test_size=0.1)
    val_df, test_df = split_dataset(val_df, test_size=0.5)

    train_df.reset_index(drop=True)
    val_df.reset_index(drop=True)
    test_df.reset_index(drop=True)

    train_df.to_json("dataset/p3_dataset_split.json", index=False)
    val_df.to_json("dataset/p3_dataset_validation.json", index=False)
    test_df.to_json("dataset/p3_dataset_test.json", index=False)

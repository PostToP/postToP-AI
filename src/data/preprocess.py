
import pandas as pd

from data.text_cleaning import DatasetPreprocessor, generate_train_preprocess_pipeline, split_dataset


def preprocess_dataset():
    dataset = pd.read_json('dataset/videos.json')

    dataset = dataset.dropna(subset=['Is Music'])

    # Initial data cleaning
    dataset["Title"] = dataset["Title"].fillna("")
    dataset["Description"] = dataset["Description"].fillna("")

    # Process text columns
    text_columns = ['Title', 'Description']
    train_preprocessor = generate_train_preprocess_pipeline(dataset)
    train_df = train_preprocessor.process_text_columns_multiprocessing(
        dataset, text_columns)
    train_df = DatasetPreprocessor.remove_similar_rows(train_df)

    train_df.to_json('dataset/p2_dataset.json', index=False)

    return train_df
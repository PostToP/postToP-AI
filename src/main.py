import numpy as np
import pickle
import dill
from tensorflow.keras.layers import Dense, Input, Dropout, concatenate
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import pandas as pd
from text_cleaning import TextPreprocessor, generate_test_preprocess_pipeline, generate_train_preprocess_pipeline, split_dataset, DatasetPreprocessor
from tokenizer import TokenizerNgram, TokenizerNone, TokenizerWord
from vectorizer import VectorizerBert, VectorizerCount, VectorizerLabel, VectorizerTFIDF


def encode_df(dataset, tokenizer, vectorizer):
    train, val, test = dataset

    tokenizer.train(train)
    train_token = [tokenizer.encode(x) for x in train]
    val_token = [tokenizer.encode(x) for x in val]
    if test is not None:
        test_token = [tokenizer.encode(x) for x in test]

    vectorizer.train(train_token)
    train_vectors = vectorizer.encode_batch(train_token)
    val_vectors = vectorizer.encode_batch(val_token)
    if test is not None:
        test_vectors = vectorizer.encode_batch(test_token)
    else:
        test_vectors = None

    return train_vectors, val_vectors, test_vectors


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

train_df, test_df = split_dataset(train_df)

# # Save processed datasets
# train_df.to_json('pp3_train_dataset.json')
# test_df.to_json('pp3_test_dataset.json')


final_df = pd.concat([train_df, test_df])
final_train_df, final_val_df = train_test_split(
    final_df, test_size=0.1, random_state=69)
final_train_df_labels = final_train_df['Is Music'].values.astype(int)
final_val_df_labels = final_val_df['Is Music'].values.astype(int)


def compile_final():
    print("Compiling model")
    title_tokenizer = TokenizerNgram((1, 3))
    title_vectorizer = VectorizerCount(8500)
    description_tokenizer = TokenizerWord()
    description_vectorizer = VectorizerTFIDF()
    category_tokenizer = TokenizerNone()
    category_vectorizer = VectorizerLabel()

    print("Encoding data")

    train_title_vectors, val_title_vectors, test_title_vectors = encode_df(
        (final_train_df['Title'], final_val_df['Title'], None), title_tokenizer, title_vectorizer)
    print("Title done")
    train_desc_vectors, val_desc_vectors, test_desc_vectors = encode_df(
        (final_train_df['Description'], final_val_df['Description'], None), description_tokenizer, description_vectorizer)
    print("Description done")
    train_cat_vectors, val_cat_vectors, test_cat_vectors = encode_df(
        (final_train_df['Categories'], final_val_df['Categories'], None), category_tokenizer, category_vectorizer)
    print("Category done")
    train_duration = final_train_df["Duration"].values.astype(int)
    val_duration = final_val_df["Duration"].values.astype(int)

    title_input = Input(shape=(train_title_vectors.shape[1],))
    title_x = Dense(4, activation='elu')(title_input)
    title_x = Dropout(0.1)(title_x)
    title_x = Dense(2, activation='sigmoid')(title_x)
    title_x = Dropout(0.2)(title_x)

    desc_input = Input(shape=(train_desc_vectors.shape[1],))
    desc_x = Dense(64, activation='relu')(desc_input)
    desc_x = Dropout(0.1)(desc_x)

    cat_input = Input(shape=(train_cat_vectors.shape[1],))
    cat_x = cat_input

    dur_input = Input(shape=(1,))
    dur_x = Dense(256, activation='sigmoid')(dur_input)
    dur_x = Dropout(0.2)(dur_x)
    dur_x = Dense(32, activation='tanh')(dur_x)
    dur_x = Dropout(0.3)(dur_x)

    combined = layers.concatenate([title_x, desc_x, cat_x, dur_x])
    x = combined

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[title_input, desc_input,
                  cat_input, dur_input], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    anti_overfit = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, min_delta=0.005, mode="min")
    model.fit(
        [train_title_vectors, train_desc_vectors,
            train_cat_vectors, train_duration],
        final_train_df_labels,
        epochs=5000,
        validation_data=([val_title_vectors, val_desc_vectors,
                         val_cat_vectors, val_duration], final_val_df_labels),
        callbacks=[anti_overfit],)

    title_pipeline = (title_tokenizer, title_vectorizer)
    description_pipeline = (description_tokenizer, description_vectorizer)
    category_pipeline = (category_tokenizer, category_vectorizer)

    return model, title_pipeline, description_pipeline, category_pipeline


model, title_pipeline, description_pipeline, category_pipeline = compile_final()


class ModelWrapper:
    def __init__(self, model, title_pipeline, description_pipeline, category_pipeline):
        self.model = model
        self.title_pipeline = title_pipeline
        self.description_pipeline = description_pipeline
        self.category_pipeline = category_pipeline
        self.text_cleaner = generate_test_preprocess_pipeline()

    def predict(self, title, description, category, duration):
        title = self.text_cleaner.process_text(title)
        description = self.text_cleaner.process_text(description)
        title = self.title_pipeline[0].encode(title)
        description = self.description_pipeline[0].encode(description)
        category = self.category_pipeline[0].encode(category)
        duration = [duration]

        title = self.title_pipeline[1].encode(title)
        description = self.description_pipeline[1].encode(description)
        category = self.category_pipeline[1].encode(category)

        title = np.array(title)
        description = np.array(description)
        category = np.array(category)
        duration = np.array(duration)

        title = title.reshape(1, -1)
        description = description.reshape(1, -1)
        category = category.reshape(1, -1)
        duration = duration

        return self.model.predict([title, description, category, duration])


with open('model.pkl', 'wb') as f:
    dill.dump(ModelWrapper(model, title_pipeline,
              description_pipeline, category_pipeline), f)

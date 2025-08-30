from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import pandas as pd
from data.text_cleaning import  split_dataset
from model.ModelWrapper import ModelWrapper
import logging
from vectorizer.VectorizerLabel import VectorizerLabel
from vectorizer.VectorizerSequential import VectorizerSequential
from tokenizer.TokenizerWhitespace import TokenizerWhitespace
from tokenizer.TokenizerNone import TokenizerNone

logger = logging.getLogger("experiment")

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

def create_model():
    train_df = pd.read_json('dataset/p2_dataset.json')
    logger.info(f"Dataset size: {len(train_df)}")
    
    final_train_df, final_val_df = split_dataset(train_df)
    final_val_df, final_test_df = split_dataset(final_val_df, test_size=0.5)
    logger.debug(f"Train size: {len(final_train_df)}, Val size: {len(final_val_df)}, Test size: {len(final_test_df)}")
    final_train_df_labels = final_train_df['Is Music'].values.astype(int)
    final_val_df_labels = final_val_df['Is Music'].values.astype(int)
    final_test_df_labels = final_test_df['Is Music'].values.astype(int)

    logger.info("Compiling model")
    title_tokenizer = TokenizerWhitespace()
    title_vectorizer = VectorizerSequential(8500,20)
    description_tokenizer = TokenizerWhitespace()
    description_vectorizer = VectorizerSequential(5000,100)
    category_tokenizer = TokenizerNone()
    category_vectorizer = VectorizerLabel()
    logger.debug(f"Title pipeline: {title_tokenizer}, {title_vectorizer}")
    logger.debug(f"Description pipeline: {description_tokenizer}, {description_vectorizer}")
    logger.debug(f"Category pipeline: {category_tokenizer}, {category_vectorizer}")


    logger.info("Encoding data")

    train_title_vectors, val_title_vectors, test_title_vectors = encode_df(
        (final_train_df['Title'], final_val_df['Title'], final_test_df['Title']), title_tokenizer, title_vectorizer)
    logger.info(f"Title done")
    logger.debug(f"Title train shape: {train_title_vectors.shape}, val shape: {val_title_vectors.shape}, test shape: {test_title_vectors.shape}")
    train_desc_vectors, val_desc_vectors, test_desc_vectors = encode_df(
        (final_train_df['Description'], final_val_df['Description'], final_test_df['Description']), description_tokenizer, description_vectorizer)
    logger.info(f"Description done")
    logger.debug(f"Description train shape: {train_desc_vectors.shape}, val shape: {val_desc_vectors.shape}, test shape: {test_desc_vectors.shape}")
    train_cat_vectors, val_cat_vectors, test_cat_vectors = encode_df(
        (final_train_df['Categories'], final_val_df['Categories'], final_test_df['Categories']), category_tokenizer, category_vectorizer)
    logger.info(f"Category done")
    logger.debug(f"Category train shape: {train_cat_vectors.shape}, val shape: {val_cat_vectors.shape}, test shape: {test_cat_vectors.shape}")
    train_duration = final_train_df["Duration"].values.astype(int)
    val_duration = final_val_df["Duration"].values.astype(int)
    test_duration = final_test_df["Duration"].values.astype(int)

    title_input = Input(shape=(train_title_vectors.shape[1],), name="title_input")
    title_x = Embedding(input_dim=8500, output_dim=12)(title_input)
    title_x = GlobalAveragePooling1D()(title_x)
    title_x = Dense(8, activation="elu")(title_x)
    title_x = Dropout(0.2)(title_x)

    desc_input = Input(shape=(train_desc_vectors.shape[1],), name="desc_input")
    desc_x = Embedding(input_dim=5000, output_dim=16)(desc_input)
    desc_x = GlobalAveragePooling1D()(desc_x)
    desc_x = Dense(64, activation='relu')(desc_x)
    desc_x = Dropout(0.1)(desc_x)

    cat_input = Input(shape=(train_cat_vectors.shape[1],), name="cat_input")
    cat_x = cat_input

    dur_input = Input(shape=(1,), name="dur_input")
    dur_x = Dense(8, activation='sigmoid')(dur_input)
    dur_x = Dropout(0.2)(dur_x)
    dur_x = Dense(4, activation='tanh')(dur_x)
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
        monitor='val_loss', patience=40, restore_best_weights=True, mode="min", min_delta=0.001)
    model.fit(
        [train_title_vectors, train_desc_vectors,
            train_cat_vectors, train_duration],
        final_train_df_labels,
        epochs=5000,
        batch_size=512,
        validation_data=([val_title_vectors, val_desc_vectors,
                        val_cat_vectors, val_duration], final_val_df_labels),
        callbacks=[anti_overfit],)
    
    loss, acc = model.evaluate(
        [test_title_vectors, test_desc_vectors, test_cat_vectors, test_duration], final_test_df_labels, verbose=2)
    logger.info(f"Validation accuracy: {acc}, loss: {loss}")

    title_pipeline = (title_tokenizer, title_vectorizer)
    description_pipeline = (description_tokenizer, description_vectorizer)
    category_pipeline = (category_tokenizer, category_vectorizer)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    model_wrapper = ModelWrapper(title_pipeline, description_pipeline, category_pipeline)
    model_wrapper.save_model(tflite_model, 'model/model.tflite')
    model_wrapper.serialize('model/v1.pkl')

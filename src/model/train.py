import logging

import optuna
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import FunctionTransformer, Pipeline
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
)
from tensorflow.keras.models import Model

from data.text_cleaning import split_dataset
from model.model_wrapper import ModelWrapper
from vectorizer.vectorizer_bucket import VectorizerBucket
from tokenizer.tokenizer_whitespace import TokenizerWhitespace
from vectorizer.vectorizer_label import VectorizerLabel
from vectorizer.vectorizer_sequential import VectorizerSequential

logger = logging.getLogger("experiment")


def build_model(params: dict) -> Model:
    inputs = []
    valami = []
    paths = ["title", "desc", "cat", "dur"]

    for path in paths:
        inputs.append(Input(shape=(params[f"{path}_input_dim"],), name=f"{path}_input"))
        x = inputs[-1]
        if f"{path}_embed_dim" in params:
            x = layers.Embedding(
                input_dim=params[f"{path}_vocab_size"],
                output_dim=2 ** params[f"{path}_embed_dim"],
            )(inputs[-1])
            x = layers.GlobalAveragePooling1D()(x)
        n_layers = params[f"n_{path}_layers"]
        for i in range(n_layers):
            x = layers.Dense(
                2 ** params[f"{path}_units_{i}"],
                activation=params[f"{path}_activation_{i}"],
                kernel_regularizer=regularizers.l2(params[f"{path}_l2_{i}"]),
            )(x)
            x = layers.Dropout(params[f"{path}_dropout_{i}"])(x)
        valami.append(x)

    combined = layers.concatenate(valami)
    x = combined

    # --- Post-processing layers ---
    n_post_layers = params["n_post_layers"]
    for i in range(n_post_layers):
        x = layers.Dense(
            params[f"post_units_{i}"],
            activation=params[f"post_activation_{i}"],
            kernel_regularizer=regularizers.l2(params[f"post_l2_{i}"]),
        )(x)
        x = layers.Dropout(params[f"post_dropout_{i}"])(x)

    output = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def train_and_evaluate(
    input_frames: list,
    pipelines: list,
    model_params: dict,
) -> tuple[Model, float, float]:
    dataset = pd.concat(input_frames, ignore_index=True, axis=1)
    train_df, val_df = split_dataset(dataset, test_size=0.1)
    val_df, test_df = split_dataset(val_df, test_size=0.5)

    train_inputs = []
    val_inputs = []
    test_inputs = []

    for i in range(len(input_frames) - 1):
        pipeline = pipelines[i]
        train_input = pipeline.fit_transform(train_df[i])
        val_input = pipeline.transform(val_df[i])
        test_input = pipeline.transform(test_df[i])

        train_inputs.append(train_input)
        val_inputs.append(val_input)
        test_inputs.append(test_input)
        logger.info(f"Pipeline {i} done")
        logger.debug(
            f"Pipeline {i} train shape: {train_input.shape}, val shape: {val_input.shape}, test shape: {test_input.shape}",
        )

    train_labels = train_df[len(input_frames) - 1].to_numpy(dtype=int)
    val_labels = val_df[len(input_frames) - 1].to_numpy(dtype=int)
    test_labels = test_df[len(input_frames) - 1].to_numpy(dtype=int)

    model_params["title_input_dim"] = train_inputs[0].shape[1]
    model_params["desc_input_dim"] = train_inputs[1].shape[1]
    model_params["cat_input_dim"] = train_inputs[2].shape[1]
    model_params["dur_input_dim"] = 1 if len(train_inputs[3].shape) == 1 else train_inputs[3].shape[1]

    model = build_model(model_params)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-5,
        clipnorm=1.0,
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1,
    )
    anti_overfit = EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        mode="min",
        min_delta=0.0001,
    )
    model.fit(
        train_inputs,
        train_labels,
        epochs=5000,
        batch_size=1024,
        validation_data=(val_inputs, val_labels),
        callbacks=[anti_overfit, reduce_lr],
        shuffle=True,
    )

    loss, acc = model.evaluate(test_inputs, test_labels, verbose=2)
    logger.info(f"Validation accuracy: {acc}, loss: {loss}")

    return model, loss, acc


def create_model() -> None:
    df = pd.read_json("dataset/p2_dataset.json")
    logger.info(f"Dataset size: {len(df)}")

    logger.info("Compiling model")
    title_pipeline = Pipeline(
        [
            ("tokenizer", TokenizerWhitespace()),
            ("vectorizer", VectorizerSequential(8500, 20)),
        ],
    )
    description_pipeline = Pipeline(
        [
            ("tokenizer", TokenizerWhitespace()),
            ("vectorizer", VectorizerSequential(5000, 30)),
        ],
    )
    category_pipeline = Pipeline(
        [
            ("vectorizer", VectorizerLabel()),
        ],
    )
    duration_pipeline = Pipeline(
        [
            ("nothing2", VectorizerBucket([0, 60, 180, 360, 9999999])),
        ]
    )

    model_params = {
        "title_vocab_size": 8500,
        "title_embed_dim": 20,
        "desc_vocab_size": 5000,
        "desc_embed_dim": 30,
        "dur_vocab_size": 4,
    }

    storage_name = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="hypeparam_tuning",
        directions=["minimize", "maximize"],
        storage=storage_name,
        load_if_exists=True,
    )

    best_params = study.best_trials[0].params
    model_params.update(best_params)

    model, _loss, _acc = train_and_evaluate(
        [df["Title"], df["Description"], df["Categories"], df["Duration"], df["Is Music"]],
        [title_pipeline, description_pipeline, category_pipeline, duration_pipeline],
        model_params,
    )

    logger.debug(f"Title pipeline: {title_pipeline}")
    logger.debug(f"Description pipeline: {description_pipeline}")
    logger.debug(f"Category pipeline: {category_pipeline}")

    logger.info("Encoding data")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    model_wrapper = ModelWrapper(title_pipeline, description_pipeline, category_pipeline)
    model_wrapper.save_model(tflite_model, "model/model.tflite")
    model_wrapper.serialize("model/v1.pkl")

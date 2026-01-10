import logging

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input,
)
from tensorflow.keras.models import Model

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


def load_split(path: str) -> tuple:
    data = np.load(path, allow_pickle=True)
    X = [
        data["YT_ID"],
        data["Title"],
        data["Description"],
        data["Categories"],
        data["Duration"],
    ]
    y = data["Labels"]
    return X, y


def train_and_evaluate(
    train_df: tuple,
    test_df: tuple,
    val_df: tuple,
    model_params: dict,
) -> tuple[Model, float, float]:
    train_inputs = train_df[0]
    val_inputs = val_df[0]
    test_inputs = test_df[0]

    train_labels = train_df[1]
    val_labels = val_df[1]
    test_labels = test_df[1]

    print(train_inputs[0].shape)

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


def flag_suspicious_rows(
    model: Model,
) -> pd.DataFrame:
    train_df = load_split("dataset/p4_train.npz")
    val_df = load_split("dataset/p4_val.npz")
    test_df = load_split("dataset/p4_test.npz")

    all_inputs = []
    for i in range(len(train_df[0])):
        all_inputs.append(
            np.concatenate(
                [
                    train_df[0][i],
                    val_df[0][i],
                    test_df[0][i],
                ],
                axis=0,
            )
        )
    all_labels = np.concatenate(
        [
            train_df[1],
            val_df[1],
            test_df[1],
        ],
        axis=0,
    )

    train_inputs = all_inputs[1:]  # skip YT_ID
    train_labels = all_labels

    train_preds = model.predict(train_inputs, batch_size=1024).flatten()
    train_labels_array = train_labels.astype(np.float32).flatten()

    epsilon = 1e-7
    train_preds_clipped = np.clip(train_preds, epsilon, 1 - epsilon)
    train_loss = -(
        train_labels_array * np.log(train_preds_clipped) + (1 - train_labels_array) * np.log(1 - train_preds_clipped)
    )

    confidence = np.abs(train_preds - 0.5) * 2
    pred_label = (train_preds > 0.5).astype(int)

    disagree = pred_label != train_labels_array
    loss_thresh = np.percentile(train_loss, 90)

    suspect_mask = (train_loss > loss_thresh) & (confidence > 0.8) & disagree
    suspect_row_ids = np.where(suspect_mask)[0]
    logger.warning(f"Flagged {len(suspect_row_ids)} suspicious rows")

    suspect_rows = all_inputs[0][suspect_row_ids]
    original_dataset = pd.read_json("dataset/videos.json")
    yt_id_to_index = {yt_id: idx for idx, yt_id in enumerate(original_dataset["YT ID"])}
    suspect_rows = [yt_id_to_index[yt_id] for yt_id in suspect_rows if yt_id in yt_id_to_index]

    suspect_rows_df = original_dataset.iloc[suspect_rows]
    suspect_rows_df = suspect_rows_df.assign(
        Predicted_Probability=train_preds[suspect_row_ids],
        Predicted_Label=pred_label[suspect_row_ids],
        Loss=train_loss[suspect_row_ids],
    )
    suspect_rows_df = suspect_rows_df.sort_values(by="Loss", ascending=False)
    suspect_rows_df.to_csv("model/suspect_rows.csv", index=False)


def create_model() -> None:
    train_df = load_split("dataset/p4_train.npz")
    val_df = load_split("dataset/p4_val.npz")
    test_df = load_split("dataset/p4_test.npz")

    train_df = (train_df[0][1:], train_df[1])
    val_df = (val_df[0][1:], val_df[1])
    test_df = (test_df[0][1:], test_df[1])

    logger.info("Compiling model")

    model_params = {
        "title_vocab_size": 8500,
        "title_embed_dim": 20,
        "desc_vocab_size": 5000,
        "desc_embed_dim": 30,
        "dur_vocab_size": 4,
    }

    storage_name = "sqlite:///optuna_study.db"
    study = optuna.load_study(
        study_name="hypeparam_tuning",
        storage=storage_name,
    )

    best_params = study.best_trials[0].params
    model_params.update(best_params)

    model, _loss, _acc = train_and_evaluate(
        train_df,
        test_df,
        val_df,
        model_params,
    )

    flag_suspicious_rows(model)
    model.summary()

    model.save("model/final_model.keras")

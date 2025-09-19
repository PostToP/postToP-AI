from functools import lru_cache

import optuna
import pandas as pd
import tensorflow as tf
from sklearn.pipeline import FunctionTransformer, Pipeline
from tensorflow.keras import Input, Model, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from data.text_cleaning import split_dataset
from tokenizer.tokenizer_whitespace import TokenizerWhitespace
from vectorizer.vectorizer_label import VectorizerLabel
from vectorizer.vectorizer_sequential import VectorizerSequential


def generate_hyperparameters(trial: optuna.Trial) -> dict:
    params = {
        "title_embed_dim": 2 ** trial.suggest_int("title_embed_dim", 3, 6),
        "desc_embed_dim": 2 ** trial.suggest_int("desc_embed_dim", 4, 7),
        "learning_rate": 2e-4,
    }
    paths = ["title", "desc", "cat", "dur", "post"]
    for path in paths:
        n_layers = trial.suggest_int(f"n_{path}_layers", 0, 3)
        params[f"n_{path}_layers"] = n_layers
        for i in range(n_layers):
            params[f"{path}_units_{i}"] = 2 ** trial.suggest_int(f"{path}_units_{i}", 3, 8)
            params[f"{path}_activation_{i}"] = trial.suggest_categorical(
                f"{path}_activation_{i}",
                ["relu", "elu", "gelu", "tanh", "sigmoid"],
            )
            params[f"{path}_l2_{i}"] = trial.suggest_float(f"{path}_l2_{i}", 1e-7, 1e-3, log=True)
            params[f"{path}_dropout_{i}"] = trial.suggest_float(f"{path}_dropout_{i}", 0.0, 0.5)

    return params


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
                output_dim=params[f"{path}_embed_dim"],
            )(inputs[-1])
            x = layers.GlobalAveragePooling1D()(x)
        n_layers = params[f"n_{path}_layers"]
        for i in range(n_layers):
            x = layers.Dense(
                params[f"{path}_units_{i}"],
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


# yank idc
@lru_cache(maxsize=1)
def get_stuff():
    df = pd.read_json("dataset/p2_dataset.json")

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
    duration_pipeline = Pipeline([("nothing", FunctionTransformer(lambda x: x))])

    model_params = {
        "title_vocab_size": 8500,
        "title_embed_dim": 20,
        "desc_vocab_size": 5000,
        "desc_embed_dim": 30,
    }
    input_frames = [df["Title"], df["Description"], df["Categories"], df["Duration"], df["Is Music"]]
    pipelines: list[Pipeline] = [title_pipeline, description_pipeline, category_pipeline, duration_pipeline]
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

    train_labels = train_df[len(input_frames) - 1].to_numpy(dtype=int)
    val_labels = val_df[len(input_frames) - 1].to_numpy(dtype=int)
    test_labels = test_df[len(input_frames) - 1].to_numpy(dtype=int)

    model_params["title_input_dim"] = train_inputs[0].shape[1]
    model_params["desc_input_dim"] = train_inputs[1].shape[1]
    model_params["cat_input_dim"] = train_inputs[2].shape[1]
    model_params["dur_input_dim"] = 1 if len(train_inputs[3].shape) == 1 else train_inputs[3].shape[1]

    return model_params, train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels


def objective(trial: optuna.Trial) -> float:
    model_params, train_inputs, val_inputs, test_inputs, train_labels, val_labels, test_labels = get_stuff()
    params = generate_hyperparameters(trial)
    model_params.update(params)
    model = build_model(model_params)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_params["learning_rate"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    model.fit(
        train_inputs,
        train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=5000,
        batch_size=256,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    test_loss, test_acc = model.evaluate(test_inputs, test_labels, verbose=0)

    return test_loss, test_acc


def main() -> None:
    storage_name = "sqlite:///optuna_study.db"
    study = optuna.create_study(
        study_name="hypeparam_tuning",
        directions=["minimize", "maximize"],
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
    print("Best trial:")
    trials = study.best_trials
    for trial in trials:
        print(f"Trial number: {trial.number}")
        print(f"  Value: {trial.values}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

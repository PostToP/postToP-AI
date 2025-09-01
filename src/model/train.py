from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding, GlobalAveragePooling1D
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
import pandas as pd
from data.text_cleaning import  split_dataset
from model.ModelWrapper import ModelWrapper
import logging
from model.Pipeline import Pipeline
from vectorizer.VectorizerLabel import VectorizerLabel
from vectorizer.VectorizerSequential import VectorizerSequential
from tokenizer.TokenizerWhitespace import TokenizerWhitespace

logger = logging.getLogger("experiment")


def modell(params):
    title_input = Input(shape=(params['title_input_dim'],), name="title_input")
    title_x = Embedding(input_dim=params['title_vocab_size'], output_dim=params['title_embed_dim'])(title_input)
    title_x = GlobalAveragePooling1D()(title_x)
    title_x = Dense(8, activation="elu")(title_x)
    title_x = Dropout(0.2)(title_x)

    desc_input = Input(shape=(params['desc_input_dim'],), name="desc_input")
    desc_x = Embedding(input_dim=params['desc_vocab_size'], output_dim=params['desc_embed_dim'])(desc_input)
    desc_x = GlobalAveragePooling1D()(desc_x)
    desc_x = Dense(64, activation='relu')(desc_x)
    desc_x = Dropout(0.1)(desc_x)

    cat_input = Input(shape=(params['cat_input_dim'],), name="cat_input")
    cat_x = cat_input

    dur_input = Input(shape=(params['dur_input_dim'],), name="dur_input")
    dur_x = Dense(8, activation='sigmoid')(dur_input)
    dur_x = Dropout(0.2)(dur_x)
    dur_x = Dense(4, activation='tanh')(dur_x)
    dur_x = Dropout(0.3)(dur_x)

    combined = layers.concatenate([title_x, desc_x, cat_x, dur_x])
    x = combined

    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[title_input, desc_input,
                cat_input, dur_input], outputs=output)
    return model

def train_and_evaluate(input_frames, pipelines, model_params):
    dataset = pd.concat(input_frames, ignore_index=True,axis=1)
    train_df, val_df = split_dataset(dataset, test_size=0.1)
    val_df, test_df = split_dataset(val_df, test_size=0.5)
    
    
    train_inputs = []
    val_inputs = []
    test_inputs = []

    for i in range(len(input_frames)-1):
        pipeline = pipelines[i]
        train_input, val_input, test_input = pipeline.train_and_process(train_df[i], val_df[i], test_df[i])
        train_inputs.append(train_input)
        val_inputs.append(val_input)
        test_inputs.append(test_input)
        logger.info(f"Pipeline {i} done")
        logger.debug(f"Pipeline {i} train shape: {train_input.shape}, val shape: {val_input.shape}, test shape: {test_input.shape}")

    train_labels = train_df[len(input_frames)-1].values.astype(int)
    val_labels = val_df[len(input_frames)-1].values.astype(int)
    test_labels = test_df[len(input_frames)-1].values.astype(int)

    model_params['title_input_dim'] = train_inputs[0].shape[1]
    model_params['desc_input_dim'] = train_inputs[1].shape[1]
    model_params['cat_input_dim'] = train_inputs[2].shape[1]
    model_params['dur_input_dim'] = 1 if len(train_inputs[3].shape) == 1 else train_inputs[3].shape[1]

    model = modell(model_params)

    optimizer = tf.keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    anti_overfit = EarlyStopping(
        monitor='val_loss', patience=40, restore_best_weights=True, mode="min", min_delta=0.001)
    model.fit(
        train_inputs,
        train_labels,
        epochs=5000,
        batch_size=512,
        validation_data=(val_inputs, val_labels),
        callbacks=[anti_overfit],)
    
    loss, acc = model.evaluate(
        test_inputs, test_labels, verbose=2)
    logger.info(f"Validation accuracy: {acc}, loss: {loss}")

    return model, loss, acc
        

    
 

def create_model():
    df = pd.read_json('dataset/p2_dataset.json')
    logger.info(f"Dataset size: {len(df)}")

    logger.info("Compiling model")
    title_pipeline = Pipeline()
    title_pipeline.set_tokenizer(TokenizerWhitespace())
    title_pipeline.set_vectorizer(VectorizerSequential(8500,20))
    description_pipeline = Pipeline()
    description_pipeline.set_tokenizer(TokenizerWhitespace())
    description_pipeline.set_vectorizer(VectorizerSequential(5000,100))
    category_pipeline = Pipeline()
    category_pipeline.set_tokenizer(None)
    category_pipeline.set_vectorizer(VectorizerLabel())
    category_pipeline.train(df['Categories'])
    duration_pipeline = Pipeline()
    duration_pipeline.set_tokenizer(None)
    duration_pipeline.set_vectorizer(None)

    model_params = {
        'title_vocab_size': 8500,
        'title_embed_dim': 20,
        'desc_vocab_size': 5000,
        'desc_embed_dim': 30,
    }

    model, loss, acc = train_and_evaluate([df['Title'], df['Description'], df['Categories'], df['Duration'], df['Is Music']],
                         [title_pipeline, description_pipeline, category_pipeline,duration_pipeline], model_params)


    logger.debug(f"Title pipeline: {title_pipeline}")
    logger.debug(f"Description pipeline: {description_pipeline}")
    logger.debug(f"Category pipeline: {category_pipeline}")


    logger.info("Encoding data")


    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    model_wrapper = ModelWrapper(title_pipeline, description_pipeline, category_pipeline)
    model_wrapper.save_model(tflite_model, 'model/model.tflite')
    model_wrapper.serialize('model/v1.pkl')

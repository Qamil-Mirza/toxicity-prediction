import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from model import ToxicityPredictionModel
from loss_functions import masked_loss
from config import Config
from utils import load_data

def train():
    x_train, y_train, y_train_mask = load_data()
    num_features = x_train.shape[1]
    num_tasks = y_train.shape[1]

    # Create the model
    model = ToxicityPredictionModel(num_features, num_tasks, Config.NUM_HIDDEN_UNITS, Config.DROPOUT_RATE)

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss=masked_loss,
        metrics=['accuracy']
    )

    # fit the model
    history = model.fit(x_train, y_train, epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, verbose=1)

    return history, model
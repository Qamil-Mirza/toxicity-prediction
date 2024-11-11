import pandas as pd
from config import Config

# Load the training data
def load_data():
    x_train = pd.read_csv(Config.X_TRAIN_FILE)
    y_train = pd.read_csv(Config.Y_TRAIN_FILE)
    
    compound_ids = x_train.iloc[:, 0]
    x_train = x_train.iloc[:, 1:]
    y_train = y_train.iloc[:, 1:]

    y_train_mask = ~y_train.isna()
    y_train = y_train.fillna(0)

    return x_train.values, y_train.values, y_train_mask.values
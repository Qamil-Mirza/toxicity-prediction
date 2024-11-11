from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from loss_functions import masked_loss
from model import ToxicityPredictionModel
import pandas as pd
from config import Config

def evaluate_model(model_path):
    with custom_object_scope({'ToxicityPredictionModel': ToxicityPredictionModel, 'custom_loss': masked_loss}):
        model = load_model(model_path)
    
    # Load the data
    x_test = pd.read_csv(Config.X_TEST_FILE)
    y_test = pd.read_csv(Config.Y_TEST_FILE)

    # preprocess
    x_test = x_test.iloc[:, 1:]
    y_test = y_test.iloc[:, 1:]

    # impute missing values with median
    y_test = y_test.fillna(y_test.median())

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

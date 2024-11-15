import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from utils.loggers import setup_logger
import os

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to config.yaml
config_path = os.path.join(project_root, "config.yaml")

# log path
log_path = os.path.join(project_root, "logs/data_processing.log")

# === LOGGER === #
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(project_root, "logs/data_processing.log")
data_logger = setup_logger("data_proc_logger", log_path)

# === CONFIG === #
with open(config_path) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


# === DATA PROCESSING LOGIC === #
def fit_imputer_and_scaler(X_train, scaler, imputer):
    data_logger.info("Fitting Imputer and Scaler on Training Data")
    data_logger.info("Fitting Scaler")
    scaler.fit(X_train)
    data_logger.info("Fitting Imputer")
    imputer.fit(X_train)
    return imputer, scaler

def train_load_and_process_data():
    x_train_path = os.path.join(base_dir, config['paths']['x_train_dense'])
    y_train_path = os.path.join(base_dir, config['paths']['y_train_dense'])

    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    data_logger.info("Successfully Loaded Training Data")
    data_logger.info(f"X Train Dimensions: {x_train.shape}")
    data_logger.info(f"Y Train Dimensions: {y_train.shape}")
    data_logger.info("Now merging X train and Y train for train-validation splitting")

    df_train = pd.merge(x_train, y_train, on='Unnamed: 0', how='inner')

    data_logger.info("Successfully Merged X Train and Y Train")
    data_logger.info(f"Train DataFrame Dimensions: {df_train.shape}")

    df_cv, df_final_train = train_test_split(df_train, test_size=0.8, random_state=config['cross_validation']['random_state'])

    data_logger.info("Successfully Split Train DataFrame into Cross Validation and Final Train DataFrames")
    data_logger.info(f"CV DataFrame Dimensions: {df_cv.shape}")
    data_logger.info(f"Final Train DataFrame Dimensions: {df_final_train.shape}")
    data_logger.info("Calculating Class Density For Cross Validation -- Keeping only compounds with at least 8 labels in cross validation set")

    df_cv['label_density'] = df_cv.iloc[:, -12:].notna().sum(axis=1)
    df_cv_filtered = df_cv[df_cv['label_density'] >= 8]

    data_logger.info("Successfully Filtered Cross Validation DataFrame")
    data_logger.info("Concatening unused rows from cross validation set to final train set")

    df_final_train = pd.concat([df_final_train, df_cv[df_cv['label_density'] < 8]])

    data_logger.info("Successfully Concatenated Unused Rows from Cross Validation Set to Final Train Set")
    data_logger.info(f"Final Train DataFrame Dimensions: {df_final_train.shape}")
    data_logger.info(f"Filtered CV DataFrame Dimensions: {df_cv_filtered.shape}")

    return df_cv_filtered, df_final_train

def test_load_and_process_data(imputer, scaler):
    x_test_path = os.path.join(base_dir, config['paths']['x_test_dense'])
    y_test_path = os.path.join(base_dir, config['paths']['y_test_dense'])

    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    X_test = x_test.iloc[:, 1:]
    y_test = y_test.iloc[:, 1:]

    data_logger.info("Successfully Loaded Test Data")
    data_logger.info(f"X Test Dimensions: {x_test.shape}")
    data_logger.info(f"Y Test Dimensions: {y_test.shape}")
    data_logger.info("Imputing and Scaling Test Data")

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)

    data_logger.info("Successfully Imputed and Scaled Test Data")
    data_logger.info("Generating Masks For Missing Labels in Test Data")

    mask = ~y_test.isnull()
    y_test_filled = y_test.fillna(0)

    data_logger.info("Successfully Generated Masks For Missing Labels in Test Data")

    return X_test, y_test_filled, mask

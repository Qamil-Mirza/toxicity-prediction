from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import yaml
import joblib
import pandas as pd
import torch
from torch.optim import Adam
from models.MultiTaskToxicityModel import MultiTaskToxicityModel
from train import train_model
from utils.processData import train_load_and_process_data
import os
from utils.loggers import setup_logger

# === LOGGER === #
# === LOGGER === #
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(project_root, "src/logs/cross_validation.log")
cross_val_logger = setup_logger("cross_validation_logger", log_path)

# Load the config file
with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

cross_val_logger.info("Successfully Loaded Config File")
cross_val_logger.info("Beginning Cross-Validation With The Following Hyperparameters:")
cross_val_logger.info(f"hidden_dim: {config['hyperparameters']['hidden_dim']}")
cross_val_logger.info(f"learning_rate: {config['hyperparameters']['learning_rate']}")
cross_val_logger.info(f"batch_size: {config['hyperparameters']['batch_size']}")
cross_val_logger.info(f"epochs: {config['hyperparameters']['epochs']}")
cross_val_logger.info(f"imputation_strategy: {config['data']['imputation_strategy']}")
cross_val_logger.info(f"dropout: {config['hyperparameters']['dropout']}")

kf = KFold(n_splits=config["cross_validation"]["n_splits"], shuffle=True, random_state=42)

cross_val_logger.info("Successfully Initialized KFold Cross-Validation with the following configuration:")
cross_val_logger.info(f"n_splits: {config['cross_validation']['n_splits']}")
cross_val_logger.info(f"random_state: {config['cross_validation']['random_state']}")
cross_val_logger.info("Loading Imputer and Scaler...")

IMPUTER = joblib.load("models/imputer.pkl")
SCALER = joblib.load("models/scaler.pkl")

cross_val_logger.info("Successfully Loaded Imputer and Scaler")
cross_val_logger.info("Loading and Processing Data...")

df_cv_filtered, df_final_train = train_load_and_process_data()

cv_results = []

cross_val_logger.info("Successfully Loaded and Processed Data")
cross_val_logger.info("Starting Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(df_cv_filtered)):
    fold_train_set = df_cv_filtered.iloc[train_idx]
    fold_val_set = df_cv_filtered.iloc[val_idx]

    # Combine fold train set with final train set
    combined_train_set = pd.concat([df_final_train, fold_train_set], axis=0)

    # Separate features and labels
    X_train = combined_train_set.iloc[:, 1:-13]
    y_train = combined_train_set.iloc[:, -13:]
    X_val = fold_val_set.iloc[:, 1:-13]
    y_val = fold_val_set.iloc[:, -13:]

    # Impute feature vectors via median imputation
    X_train = IMPUTER.fit_transform(X_train)
    X_val = IMPUTER.transform(X_val)

    # Standardize features
    X_train = SCALER.fit_transform(X_train)  # This converts to numpy array
    X_val = SCALER.transform(X_val)  # This converts to numpy array

    # Drop last column of y_train and y_val which is the label_density
    y_train = y_train.iloc[:, :-1]
    y_val = y_val.iloc[:, :-1].fillna(y_val.median())

    # Reset indices for alignment
    y_val.reset_index(drop=True, inplace=True)

    # Initialize model
    model = MultiTaskToxicityModel(input_dim=X_train.shape[1])

    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=config["hyperparameters"]["learning_rate"])

    # Train model
    model, losses = train_model(model, optimizer, X_train, y_train, config["hyperparameters"]["batch_size"])

    # CROSS-VALIDATION
    fold_results = {
        "fold": fold + 1,
        "hidden_dim": config["hyperparameters"]["hidden_dim"],
        "learning_rate": config["hyperparameters"]["learning_rate"],
        "batch_size": config["hyperparameters"]["batch_size"],
        "dropout": config["hyperparameters"]["dropout"]
    }

    for task_idx, task in enumerate(y_train.columns):  # Use task index for slicing y_pred_task
        # Get the target values for the current task
        y_val_task = y_val[task]

        # Align features (X_val) with the corresponding indices in y_val_task
        X_val_task = X_val[y_val_task.index.to_numpy(), :]  # Use NumPy-style indexing

        # Predict using the model
        y_pred_task = model(torch.tensor(X_val_task, dtype=torch.float32)).detach().numpy()

        # Extract predictions for the current task (task_idx)
        y_pred_task_specific = y_pred_task[:, task_idx]  # Select the column corresponding to the task

        # Compute AUC for the current task
        auc = roc_auc_score(y_val_task, y_pred_task_specific)
        fold_results[f"auc_{task}"] = auc

    # Store results for this fold
    cv_results.append(fold_results)
    cross_val_logger.info(f"Fold {fold + 1} AUCs: {fold_results}")

# Convert cv_results to a DataFrame and include hyperparameter columns
cv_results_df = pd.DataFrame(cv_results)

# Save the cross-validation results along with hyperparameters
output_path = "results/cv_results_with_hyperparams.csv"
cv_results_df.to_csv(output_path, index=False)
cross_val_logger.info(f"Cross-Validation Results Saved to {output_path}")
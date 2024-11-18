from models.MultiTaskToxicityModel import MultiTaskToxicityModel
from utils.processData import train_load_and_process_data, fit_imputer_and_scaler
from torch.optim import Adam
from train import train_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yaml
import torch
import joblib
import os
from utils.loggers import setup_logger
from utils.visualize import plot_model_loss

# === LOGGER === #
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(project_root, "src/logs/training.log")
train_logger = setup_logger("training_logger", log_path)
print(f"Handlers for {train_logger.name}: {train_logger.handlers}")

# === CONFIG === #
with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

train_logger.info("Successfully Loaded Config File")

df_cv_filtered, df_final_train = train_load_and_process_data()

# Combine the entire dataset for final training
X_final_train = df_final_train.iloc[:, 1:-13]
y_final_train = df_final_train.iloc[:, -13:]

# drop label_density column
y_final_train = y_final_train.iloc[:, :-1]

train_logger.info("Successfully Loaded and Processed Data")
train_logger.info("Starting Final Training with the hyperparameters:")
train_logger.info(f"hidden_dim: {config['hyperparameters']['hidden_dim']}")
train_logger.info(f"learning_rate: {config['hyperparameters']['learning_rate']}")
train_logger.info(f"batch_size: {config['hyperparameters']['batch_size']}")
train_logger.info(f"epochs: {config['hyperparameters']['epochs']}")
train_logger.info(f"imputation_strategy: {config['data']['imputation_strategy']}")
train_logger.info(f"dropout: {config['hyperparameters']['dropout']}")

model = MultiTaskToxicityModel(input_dim=X_final_train.shape[1])
optimizer = Adam(model.parameters(), lr=config["hyperparameters"]["learning_rate"])

train_logger.info("Successfully Initialized Model and Optimizer")
train_logger.info("Fitting Imputer and Scaler on Training Data")

# Initialize scaler and imputer
scaler = StandardScaler()
imputer = SimpleImputer(strategy=config['data']["imputation_strategy"])

SCALER, IMPUTER = fit_imputer_and_scaler(X_final_train, scaler, imputer)

train_logger.info("Successfully Fitted Imputer and Scaler on Training Data")
train_logger.info("Saving the scaler and imputer...")

# Save the scaler and imputer
joblib.dump(SCALER, "models/scaler.pkl")
joblib.dump(IMPUTER, "models/imputer.pkl")

train_logger.info("Successfully Saved the Scaler and Imputer")
train_logger.info("Scaling and Imputing Training Data...")

# scale features
X_final_train = SCALER.transform(X_final_train)

# impute missing values
X_final_train = IMPUTER.transform(X_final_train)

train_logger.info("Successfully Scaled and Imputed Training Data")
train_logger.info("Training Final Model...")

# Train final model with optimized parameters
final_model, final_epoch_losses = train_model(model, optimizer, X_final_train, y_final_train, 
                                              config["hyperparameters"]["batch_size"]
                                              )

# TODO: VISUALIZE TRAINING LOSS and SAVE TO PLOT
plot_model_loss(final_epoch_losses, "final-model-loss.png")

# SAVE THE MODEL
train_logger.info("Successfully Trained Final Model")
train_logger.info("Saving Final Model...")
torch.save(final_model.state_dict(), "multi_task_toxicity_model.pth")

train_logger.info("Successfully Saved Final Model")
train_logger.info("Training Completed")

from models.MultiTaskToxicityModel import MultiTaskToxicityModel
from utils.processData import train_load_and_process_data, fit_imputer_and_scaler
from torch.optim import Adam
from train import train_model
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import yaml
import torch
import joblib

# === CONFIG === #
with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

df_cv_filtered, df_final_train = train_load_and_process_data()

# Combine the entire dataset for final training
X_final_train = df_final_train.iloc[:, 1:-13]
y_final_train = df_final_train.iloc[:, -13:]

# drop label_density column
y_final_train = y_final_train.iloc[:, :-1]

print('Final train data shape:', X_final_train.shape)
print('Final train labels shape:', y_final_train.shape)

model = MultiTaskToxicityModel(input_dim=X_final_train.shape[1])
optimizer = Adam(model.parameters(), lr=config["hyperparameters"]["learning_rate"])

# Initialize scaler and imputer
scaler = StandardScaler()
imputer = SimpleImputer(strategy=config['data']["imputation_strategy"])

SCALER, IMPUTER = fit_imputer_and_scaler(X_final_train, scaler, imputer)

# Save the scaler and imputer
joblib.dump(SCALER, "models/scaler.pkl")
joblib.dump(IMPUTER, "models/imputer.pkl")

# scale features
X_final_train = SCALER.transform(X_final_train)

# impute missing values
X_final_train = IMPUTER.transform(X_final_train)

# Train final model with optimized parameters
final_model, final_epoch_losses = train_model(model, optimizer, X_final_train, y_final_train, 
                                              config["hyperparameters"]["batch_size"]
                                              )

# SAVE PLOT LOSS

# SAVE THE MODEL
torch.save(final_model.state_dict(), "multi_task_toxicity_model.pth")
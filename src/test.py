import numpy as np
import pandas as pd
import yaml
import torch
from utils.processData import test_load_and_process_data
from models.MultiTaskToxicityModel import MultiTaskToxicityModel
from sklearn.metrics import roc_auc_score
import joblib
from utils.loggers import setup_logger
import os
from utils.visualize import plot_test_mean_auc
# === LOGGER === #
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path = os.path.join(project_root, "src/logs/testing.log")
test_logger = setup_logger("testing_logger", log_path)

with open("config.yaml") as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Load the scaler and imputer
SCALER = joblib.load("models/scaler.pkl")
IMPUTER = joblib.load("models/imputer.pkl")

X_test, y_test, mask = test_load_and_process_data(IMPUTER, SCALER)

# convert to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
mask_tensor = torch.tensor(mask.values, dtype=torch.float32)

# load the model
model = MultiTaskToxicityModel(input_dim=X_test.shape[1])
model.load_state_dict(torch.load("multi_task_toxicity_model.pth"))
model.eval()

# predict on test set
with torch.no_grad():
    y_test_pred = model(X_test_tensor)

# Apply the mask and evaluate
test_results = {}
for task_idx, task in enumerate(y_test.columns):
    # Extract true labels, predictions, and mask for the current task
    y_test_task = y_test_tensor[:, task_idx]
    y_test_pred_task = y_test_pred[:, task_idx]
    mask_task = mask_tensor[:, task_idx]

    # Only include valid entries in the evaluation
    valid_indices = mask_task.bool()
    y_test_task_valid = y_test_task[valid_indices]
    y_test_pred_task_valid = y_test_pred_task[valid_indices]

    # Compute AUC only if the task has at least two classes
    if len(set(y_test_task_valid.numpy())) > 1:
        auc = roc_auc_score(y_test_task_valid.numpy(), y_test_pred_task_valid.numpy())
        test_results[task] = auc
    else:
        test_logger.info(f"Task {task} has only one class; skipping AUC calculation.")

# Display results
test_logger.info("Final Test AUCs for each task:")
for task, auc in test_results.items():
    test_logger.info(f"{task}: {auc:.3f}")

final_results_df = pd.DataFrame(list(test_results.items()), columns=["Task", "AUC"])
final_results_df.to_csv("results/final_test_results.csv", index=False)

# plot the results
test_logger.info("Plotting the mean AUC on the test set...")
plot_test_mean_auc(final_results_df)

test_logger.info("Testing complete!")
test_logger.info("Results saved to results/final_test_results.csv")
test_logger.info("Plots saved to plots/test-mean-auc.png")

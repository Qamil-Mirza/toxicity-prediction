import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Construct the path to config.yaml
config_path = os.path.join(project_root, "config.yaml")


# === CONFIG === #
with open(config_path) as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

class MultiTaskToxicityModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=config["hyperparameters"]["hidden_dim"], num_tasks=12, dropout_rate=config["hyperparameters"]["dropout"]):
        super(MultiTaskToxicityModel, self).__init__()

        # Hidden layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        # dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # output layers, and we have one for each task
        self.output_layers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_tasks)])

    def forward(self, x):
        # pass through hidden layers with relu activations and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        # pass through output layers
        outputs = [torch.sigmoid(output_layer(x)) for output_layer in self.output_layers]

        # Concatenate outputs for each task along the batch dimension
        return torch.cat(outputs, dim=1)
    
# === TRAINING OBJECTIVES === #
def masked_bce_loss(y_pred, y_true, mask):
    # clamp predictions to avoid log(0)
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1 - eps)

    # compute individual binary crossentropy loss
    bce_loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
    # apply the mask
    masked_bce_loss = bce_loss * mask

    # return the mean loss
    return masked_bce_loss.sum()
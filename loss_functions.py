import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from utils import load_data

# Load the training data
x_train, y_train, y_train_mask = load_data()
num_tasks = y_train.shape[1]

# TRAINING OBJECTIVE
@tf.keras.utils.register_keras_serializable()
def masked_loss(y_true, y_pred):
    eps = 1e-7 # use this to avoid division by zero
    loss = 0
    for task_idx in range(num_tasks):
        y_t = y_true[:, task_idx]
        y_p = y_pred[:, task_idx]

        # create a mask for the current task and cast it to float32
        m = tf.cast(y_train_mask[:, task_idx], tf.float32)

        # calculate binary crossentropy for each task
        bce = binary_crossentropy(y_t, y_p)

        # apply mask to BCE element-wise
        masked_bce = tf.multiply(bce, m)

        # sum up the masked BCEs to get average loss across all tasks
        task_loss = tf.reduce_sum(masked_bce) / tf.reduce_sum(m + eps)
        loss += task_loss
    
    return loss / num_tasks

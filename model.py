import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.losses import binary_crossentropy

# MODEL ARCHITECTURE

class ToxicityPredictionModel(tf.keras.Model):
    def __init__(self, num_features, num_tasks, num_hidden_units, dropout_rate, **kwargs):
        super(ToxicityPredictionModel, self).__init__(**kwargs)
        # instance variables
        self.num_features = num_features
        self.num_tasks = num_tasks
        self.num_hidden_units = num_hidden_units
        self.dropout_rate = dropout_rate

        # define the layers
        self.hidden_layer = Dense(num_hidden_units, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.output_layer = [Dense(1, activation='sigmoid') for _ in range(num_tasks)]

    def call(self, inputs, training=False):
        x = self.hidden_layer(inputs)
        x = self.dropout(x, training=training)
        outputs = [layer(x) for layer in self.output_layer]
        return tf.concat(outputs, axis=1)
    
    # override the get_config method to enable serialization
    def get_config(self):
        # Return the configuration of the model
        config = super(ToxicityPredictionModel, self).get_config()
        config.update({
            "num_features": self.num_features,
            "num_tasks": self.num_tasks,
            "num_hidden_units": self.num_hidden_units,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
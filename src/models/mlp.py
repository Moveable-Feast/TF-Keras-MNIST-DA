from tensorflow import keras
from tensorflow.keras import layers

def create_mlp(input_dim=784, hidden_units=512, output_units=10, activation='relu', output_activation='softmax'):
    model = keras.Sequential([
        layers.Dense(hidden_units, activation=activation, input_shape=(input_dim,)),
        layers.Dense(output_units, activation=output_activation)
    ])
    return model
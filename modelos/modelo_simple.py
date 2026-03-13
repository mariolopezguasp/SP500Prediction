import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_simple_nn(input_dim, output_dim):
    """
    Returns the simplest possible neural network (minimum parameters)
    for multivariant regression.
    It is just a single Dense layer without activation function (Linear regression analogy).
    """
    model = Sequential([
        Dense(output_dim, input_shape=(input_dim,), activation=None)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

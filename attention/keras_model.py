import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, MultiHeadAttention, LSTM
from keras.optimizers import Adam

input_dim = 3
hidden_dim = 4
output_dim = 3

num_samples = 1000
seq_length = 5
epochs = 10


def get_model(input_dim, hidden_dim, output_dim):
    inputs = Input(shape=(seq_length, input_dim))
    lstm_out = LSTM(hidden_dim, return_sequences=True)(inputs)
    attn_outs = MultiHeadAttention(num_heads=1, key_dim=hidden_dim)(query = lstm_out, key = lstm_out, value = lstm_out)
    outputs = Dense(output_dim, activation="softmax")(attn_outs)

    model = Model(inputs, outputs)

    return model

def predict(model, x):
    outputs = model(x)
    outputs = tf.math.argmax(x, axis = -1)
    print(f"Outputs: {outputs}")


X = np.random.randn(num_samples, seq_length, input_dim)
y = np.random.randint(output_dim, size=(num_samples, seq_length))

model = get_model(input_dim, hidden_dim, output_dim)
model.compile(loss='sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics = ['accuracy'])

print(model.summary())

model.fit(X, y, batch_size=32, epochs=epochs)
print("Training completed.")

predict(model, np.random.randn(2, seq_length, input_dim))

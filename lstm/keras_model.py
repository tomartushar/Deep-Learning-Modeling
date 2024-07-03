import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam

input_dim = 3
hidden_dim = 5
output_dim = 3
seq_length = 5

num_samples = 1000
epochs = 10

X = np.random.randn(num_samples, seq_length, input_dim)
y = np.random.randint(output_dim, size=(num_samples, seq_length))


def get_model():
    inputs = Input(shape=(seq_length, input_dim))
    x = LSTM(hidden_dim, return_sequences=True)(inputs)
    outputs = Dense(output_dim, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model


def predict(model, x):
    outputs = model(x)
    outputs = tf.math.argmax(outputs, axis = -1)
    print(f"Outputs: ", outputs)

model = get_model()
model.compile(loss = 'sparse_categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics = ['accuracy'])

model.fit(X, y, batch_size=32, epochs = 10)
print('Training completed.')

print(predict(model, np.random.randn(2, seq_length, input_dim)))
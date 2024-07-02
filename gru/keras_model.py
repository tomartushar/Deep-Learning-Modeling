import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, GRU
from keras.optimizers import Adam


np.random.seed(0)

input_size = 3
hidden_size = 5
output_size = 3
num_epochs = 10
lr = 0.01

seq_length = 5
num_samples = 1000


X = np.random.randn(num_samples, seq_length, input_size)
y = np.random.randint(output_size, size=(num_samples, seq_length)).astype(np.int32)


def get_model():
    inputs = Input(shape=(seq_length, input_size))
    gru_out = GRU(hidden_size, return_sequences=True)(inputs)
    outputs = Dense(output_size, activation="softmax")(gru_out)
    model = Model(inputs, outputs)

    return model

def predict(model, x):
    outputs = model(x)
    outputs = tf.math.argmax(outputs, axis=-1) # for classification
    print(f"Predicted output is:\n", outputs)


model = get_model()
model.compile(loss = 'sparse_categorical_crossentropy', 
            optimizer = Adam(learning_rate = lr), 
            metrics = ['accuracy'])
print(model.summary())
model.fit(X, y, epochs = num_epochs,
        batch_size = 32, verbose = 1) # can add callbacks here like: EarlyStopping, ModelChecpuoint, LearningRateScheduler

print('training complete')
predict(model, np.random.rand(2,seq_length, input_size))

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from sklearn.model_selection import train_test_split


class BinaryClassification:
    def prepare_data(self):
        np.random.seed(42)
        self.X_train = np.random.rand(1000, 20)
        self.y_train = np.random.randint(2, size = (1000, 1))

    def model_fit(self):
        self.model = Sequential()
        self.model.add(Dense(32, input_shape = (None,20),
                        activation="relu"))
        self.model.add(Dense(16, activation = "relu"))
        self.model.add(Dense(1, activation = "sigmoid"))

        self.model.compile(optimizer = Adam(learning_rate = 0.01),
                    loss='binary_crossentropy', metrics = ['accuracy'])
        print(self.model.summary())
        self.model.fit(self.X_train, self.y_train, epochs = 20,
                        batch_size = 32, verbose = 2)
    
    def model_evaluate(self, X, y):
        loss, accuracy = self.model.evaluate(X, y)
        print(f'Loss :{loss}, Accuracy: {accuracy}')


class MultiClassificaiton:
    def prepare_data(self, num_classes, val_size = None, test_size = None, total_samples = 1000):
        self.n_classes = num_classes
        np.random.seed(42)
        tf.random.set_seed(42)
        X = np.random.rand(total_samples, 20)
        y = np.random.randint(num_classes, size=(total_samples, 1))
        y = to_categorical(y, num_classes)

        self.X_train, self.y_train = None, None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None

        if val_size:
            X_tr, self.X_test, y_tr, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                     random_state=42)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_tr, y_tr, 
                                                        test_size=(val_size/(1-test_size)),
                                                        random_state=42)
        elif test_size:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,
                                                                    y, test_size=val_size,
                                                                    random_state=42)
        else:
            self.X_train = X
            self.y_train = y        


    def model_fit(self, epochs = 100, batch_size = 32):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape = self.X_train.shape[1:]))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32))
        self.model.add(BatchNormalization())
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation="relu"))
        self.model.add(Dense(self.n_classes, activation="softmax"))

        self.model.compile(optimizer=Adam(learning_rate = 0.0001),
                      loss = "categorical_crossentropy", 
                      metrics = ['accuracy'])
        print(self.model.summary())

        print(f"Training feature set shape: {self.X_train.shape}")
        print(f"Training targt set shape: {self.y_train.shape}")
        
        early_stoppping = EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-6)
        checkpoint = ModelCheckpoint('best_model.h5', save_best_only = True, monitor = 'val_loss')
        def scheduler(epoch, lr):
            if epoch < 5:
                return lr
            else:
                return lr*tf.math.exp(-1)
        lr_scheduler = LearningRateScheduler(scheduler)

        if self.X_val is not None:
            print(f"Validation data size:- features: {self.X_val.shape}, target: {self.y_val.shape}")
            self.model.fit(self.X_train, self.y_train, epochs = epochs,
                    batch_size = batch_size, verbose = 2, validation_data=(self.X_val, self.y_val),
                    callbacks=[early_stoppping, checkpoint, lr_scheduler]
                    )
        else:
            self.model.fit(self.X_train, self.y_train, epochs = epochs,
                        batch_size = batch_size, verbose = 2,
                        callbacks=[early_stoppping, checkpoint])


    def model_evaluate(self, X, y):
        loss, accuracy = self.model.evaluate(X, y)
        print(f"loss: {loss:.6f}, accuracy: {accuracy:.3f}")


if __name__ == '__main__':
    # binary_cls = BinaryClassification()
    # print('Preparing data ...')
    # binary_cls.prepare_data()
    # print('Fitting model ....')
    # binary_cls.model_fit()
    # print('Evaluation on training set ...')
    # binary_cls.model_evaluate(binary_cls.X_train, binary_cls.y_train)

    multi_cls = MultiClassificaiton()
    print("Preparing data ...")
    multi_cls.prepare_data(3, val_size=0.15, test_size=0.15, total_samples=1500)
    print('Fitting model ...')
    multi_cls.model_fit()
    print('Model score on training set')
    multi_cls.model_evaluate(multi_cls.X_train, multi_cls.y_train)
    print('Model score on validation set')
    multi_cls.model_evaluate(multi_cls.X_val, multi_cls.y_val)
    print('Model score on test set')
    multi_cls.model_evaluate(multi_cls.X_test, multi_cls.y_test)



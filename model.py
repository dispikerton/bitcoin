import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, LSTM


class BitcoinPricePredictor:
    def __init__(self, window_size, num_features):
        self.window_size = window_size
        self.num_features = num_features
        self.model = None

    def create_model(self):
        self.model = tf.keras.Sequential([
            Input(shape=(self.window_size, self.num_features)),
            LSTM(8),
            Dense(1)
        ])
        self.model.compile(optimizer=Adam(), loss='mse')

    def train_model(self, x_train, y_train, epochs=1, batch_size=128, save_path='model.keras'):
        if self.model is None:
            self.create_model()
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        self.model.save(save_path)

    def evaluate_model(self, x_test, y_test):
        if self.model is None:
            raise Exception("Model has not been trained yet.")
        test_loss = self.model.evaluate(x_test, y_test)
        print(f'Test loss: {test_loss:.4f}')

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, x):
        if self.model is None:
            raise Exception("Model has not been trained or loaded yet.")
        return self.model.predict(x)

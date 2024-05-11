import tensorflow as tf
from tensorflow import keras

# Настройка TensorFlow для использования GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU found. Using CPU.")


def create_model(window_size, num_features):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(32, input_shape=(window_size, num_features)))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='sgd', loss='mse')
    return model


def train_model(model, x_train, y_train, epochs=50, batch_size=128):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def evaluate_model(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss:.4f}')

import tensorflow as tf


def create_model(window_size, num_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, num_features)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd', loss='mse')
    return model


def train_model(model, x_train, y_train, epochs=1, batch_size=128, save_path='model.keras'):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(save_path)


def evaluate_model(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss:.4f}')


def load_model(model_path):
    return tf.keras.models.load_model(model_path)

from tensorflow import keras


def create_model(window_size, num_features):
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, num_features)),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd', loss='mse')
    return model


def train_model(model, x_train, y_train, epochs=50, batch_size=128, save_path='model.keras'):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(save_path)


def evaluate_model(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss:.4f}')

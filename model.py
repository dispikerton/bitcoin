from keras.models import Sequential
from keras.layers import LSTM, Dense


def create_model(window_size, num_features):
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, num_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, x_train, y_train, epochs=100, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)


def evaluate_model(model, x_test, y_test):
    test_loss = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss:.4f}')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def scale_data(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data), scaler


def create_dataset(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, :])
        y.append(data[i, 3])
    return np.array(X), np.array(y)


def prepare_data(data, window_size, train_size=0.8):
    scaled_data, scaler = scale_data(data)
    train_data = scaled_data[:int(train_size * len(scaled_data))]
    test_data = scaled_data[int(train_size * len(scaled_data)):]
    x_train, y_train = create_dataset(train_data, window_size)
    x_test, y_test = create_dataset(test_data, window_size)
    return x_train, y_train, x_test, y_test, scaler

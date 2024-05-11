from data_preprocessing import load_data, prepare_data
from model import create_model, train_model, evaluate_model

file_path = 'bitcoin_2017_to_2023.csv'
window_size = 60


def main():
    data = load_data(file_path)
    print("Data loaded.")
    x_train, y_train, x_test, y_test, scaler = prepare_data(data, window_size)
    print("Data prepared.")
    model = create_model(window_size, x_train.shape[2])
    print("Model created.")
    train_model(model, x_train, y_train)
    print("Model trained.")
    evaluate_model(model, x_test, y_test)
    print("Model evaluated.")


if __name__ == '__main__':
    main()

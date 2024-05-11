from fastapi import FastAPI
import uvicorn
from data_preprocessing import load_data, prepare_data
from model import create_model, train_model, evaluate_model
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import io

app = FastAPI()

file_path = 'bitcoin_2017_to_2023.csv'
window_size = 60


@app.on_event("startup")
async def startup_event():
    global model, scaler, x_test, y_test
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


@app.get("/")
async def root():
    return {"message": "Bitcoin Price Prediction API"}


@app.get("/visualize")
async def visualize():
    predictions = model.predict(x_test)

    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
    plt.plot(scaler.inverse_transform(predictions.reshape(-1, 1)), label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

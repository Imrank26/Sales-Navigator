import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def get_realtime_data(ticker, interval='1d', range='5y'):
    data = yf.download(tickers=ticker, interval=interval, period=range)
    return data

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    return scaled_data, scaler

def create_dataset(dataset, look_back=1, days=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-days):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back:i + look_back + days, 0])
    return np.array(X), np.array(Y)

def create_transformer_model(input_shape, num_heads=2, num_layers=2, d_model=32, output_units=1):
    inputs = keras.layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_layers):
        x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = keras.layers.Flatten()(x)
    outputs = keras.layers.Dense(output_units)(x)
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_prices(model, data, scaler, look_back):
    last_data = data.tail(look_back)
    last_data_scaled = scaler.transform(last_data['Close'].values.reshape(-1, 1))
    X_test = np.array([last_data_scaled])
    prediction = model.predict(X_test)
    prediction = scaler.inverse_transform(prediction)
    return prediction[0][0]

def plot_stock_data(data):
    plt.figure(figsize=(12, 6))
    plt.title("Historical Sales Data")
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.xlabel("Date")
    plt.ylabel("Price")"c:/Users/sam/Downloads/sales.py"
    plt.legend()
    st.pyplot()
    st.set_option('deprecation.showPyplotGlobalUse', False)



def main():
    st.title("Sales Navigator")

    # Create input fields for entering the sales symbol and the number of days to predict
    symbol = st.text_input("Enter sales symbol")
    days = st.slider("Enter number of days to predict", min_value=1, max_value=30, value=1)

    # Create a button to trigger the prediction
    if st.button("Predict"):
        data = get_realtime_data(symbol)
        scaled_data, scaler = preprocess_data(data)
        X, Y = create_dataset(scaled_data, look_back=10, days=1)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        model = create_transformer_model(input_shape=(X.shape[1], 1))
        model.fit(X, Y, batch_size=64, epochs=10, verbose=1)
        predicted_price = predict_prices(model, data, scaler, look_back=10)
        st.success(f"Predicted price for {symbol} in {days} days: {predicted_price:.2f}")
        
        # Plot historical sales data
        plot_stock_data(data)
        
        # Make the prediction for the entire requested period
        predicted_data = []
        for _ in range(days):
            X_test = np.array([scaled_data[-10:]])
            prediction = model.predict(X_test)
            predicted_price = scaler.inverse_transform(prediction)[0][0]
            scaled_data = np.append(scaled_data, prediction)
            predicted_data.append(predicted_price)
        

if __name__ == "__main__":
    st.set_page_config(
        page_title="SALES NAVIGATOR",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    main()

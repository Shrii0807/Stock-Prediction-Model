import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Set the start and end dates for the data
start = '2010-01-01'
end = '2019-12-31'

# Streamlit title
st.title('Stock Trend Prediction')

# User input for stock ticker
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetching the stock data using yfinance
data = yf.download(user_input, start=start, end=end)

# Describing Data to the user
st.subheader('Data from 2010-2019')
st.write(data.describe())

# Visualizations
st.subheader('Closing Price Vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(data['Close'])
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chart with 100MA & 200MA')
ma100 = data['Close'].rolling(100).mean()
ma200 = data['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r', label='100 MA')
plt.plot(ma200, 'g', label='200 MA')
plt.plot(data['Close'], 'b', label='Close Price')
plt.legend()
st.pyplot(fig)

# Splitting data into training and testing
data_training = pd.DataFrame(data['Close'][0:int(len(data) * 0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data) * 0.70):])

print(data_training.shape)
print(data_testing.shape)

# Create an instance of the scaler
scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Loading the Model
model = load_model('my_model.keras')

# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)  # Use pd.concat instead of append
input_data = scaler.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# Scale back the predicted and test values
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

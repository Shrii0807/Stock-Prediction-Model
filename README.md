# Stock-Prediction-Model
This is a Stock Price Prediction web application that uses Long Short-Term Memory (LSTM) neural networks to forecast stock prices based on historical data. The application retrieves stock data from Yahoo Finance, processes it, and visualizes key metrics and trends. Built using Streamlit for a seamless web interface, the app provides users with an interactive experience for stock data analysis and prediction.

![image](https://github.com/user-attachments/assets/337d19c4-c8f1-4b5a-a3e8-46b904fce059)
![image](https://github.com/user-attachments/assets/d5cfdc79-60ed-4bcd-b7cc-cce3ef333d68)

# Features:

  Data Preprocessing: Efficiently handle and prepare historical stock price data for training.
  
  LSTM Model Architecture: Implements a robust LSTM architecture for stock price prediction.
  
  Model Evaluation: Evaluate model performance using metrics such as Mean Squared Error (MSE) and visualization of predictions vs actual prices.
  
  Data Visualization: Visualize stock price trends and model predictions with Matplotlib and Seaborn.

  User-Friendly Interface: A simple input field to enter the stock ticker symbol.
  
  Data Visualization: Displays a Closing Price vs. Time chart with 100-day and 200-day moving averages.
  
  Statistical Summary: Presents a table with key statistical data from 2010-2019, including open, high, low, close, and volume information.
  
  Real-Time Forecasting: Uses LSTM neural networks to predict stock prices based on historical trends.
  
# Technologies used:

Machine Learning Model: Long Short-Term Memory (LSTM) neural network

Data Source: Yahoo Finance (using yfinance library)

Web Application: Streamlit

Libraries: Pandas, Numpy, Matplotlib, TensorFlow/Keras for model building and training

 # Dataset:


The Yahoo Finance dataset typically includes historical stock market data, providing a wealth of information for financial analysis and research. 

# How It Works:

Data Extraction: The application uses the Yahoo Finance API using the stock ticker like (AAPL, SBIN.NS etc.) to extract historical stock data.

Data Processing: The data is processed to calculate 100-day and 200-day moving averages, which help smooth out short-term fluctuations and show longer-term trends.

Prediction Model: An LSTM neural network is trained on the historical data to predict future stock prices.

Streamlit Web Interface: Streamlit is used to create a simple and interactive web app where users can input a stock ticker and view predictions along with historical data and trends.

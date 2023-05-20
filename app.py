import math  # Import the math module
from tensorflow.keras.layers import LSTM  # Import the LSTM layer from TensorFlow Keras
from tensorflow.keras.layers import Dense  # Import the Dense layer from TensorFlow Keras
from tensorflow.keras.models import Sequential  # Import the Sequential model from TensorFlow Keras
from sklearn.preprocessing import MinMaxScaler  # Import the MinMaxScaler from scikit-learn
import streamlit as st  # Import the streamlit module
from datetime import date  # Import the date module from datetime
import pandas as pd  # Import the pandas library
import numpy as np  # Import the numpy library
import yfinance as yf  # Import the yfinance library for fetching stock data
import plotly.graph_objects as go  # Import the graph_objects module from plotly

# Set the title of the Streamlit app
st.title('Stock Forecast App')

# Define the list of stocks for selection
stocks = ("BHARTIARTL.NS", "ICICIBANK.NS", "TATASTEEL.NS","AAPL", "GOOGL", "MSFT","BTC-USD", "ETH-USD", "LTC-USD")
selected_stock = st.selectbox("Select Stocks for prediction", stocks)

# Function to load data using Yahoo Finance API
def load_data(ticker):
    data = yf.download(ticker)  # Download stock data using Yahoo Finance
    data.reset_index(inplace=True)  # Reset index of the data frame
    return data

# Display loading state while loading data
data_load_state = st.text('Loading data...')
df = load_data(selected_stock)  # Load stock data for the selected stock
data_load_state.text('Loading data... done!')

# Display the last five days of data
st.subheader('Last Five Days')
st.write(df.tail())

# Function to plot the raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.Date, y=df['Close'], name="stock_close", line_color='deepskyblue'))
    fig.layout.update(
        title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    return fig

# Display the raw data plot
plot_raw_data()

# Display loading state while loading the model
data_load_state = st.text('Loading Model...')

# Extract the 'Close' column for modeling
data = df.filter(['Close'])
current_data = np.array(data).reshape(-1, 1).tolist()

# Reshape and scale the data
df = np.array(data).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(np.array(df).reshape(-1, 1))
train_data = scaled_df[0:, :]

x_train = []
y_train = []
duration = 90

# Create input sequences and target variables for training
for i in range(duration, len(train_data)):
    x_train.append(train_data[i-duration:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=2)

# Prepare test data for prediction
test_data = scaled_df[-duration:, :].tolist()
x_test = []
y_test = []
for i in range(duration, duration+10):
    x_test = (test_data[i-duration:i])
    x_test = np.asarray(x_test)
    pred_data = model.predict(x_test.reshape(1, x_test.shape[0], 1).tolist())

    y_test.append(pred_data[0][0])
    test_data.append(pred_data)

# Inverse scale the predicted values
pred_next_10 = scaler.inverse_transform(np.asarray(y_test).reshape(-1, 1))

# Evaluate model accuracy
train_loss = model.evaluate(x_train, y_train, verbose=0)
model_accuracy = 1 - train_loss

data_load_state.text('Loading Model... done!')

# Display the model accuracy
st.subheader("Model Accuracy")
st.write("Accuracy:", model_accuracy)

# Display the predicted values for the next 10 days
st.subheader("Next 10 Days")
st.write(pred_next_10)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Step 1: Data Preprocessing
data = pd.read_csv('/content/series_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# Step 2: Feature Engineering
# You can add more features if needed, like day of the week, month, etc.

# Step 3: Data Normalization
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Step 4: Train-Validation Split
train_size = int(len(data_scaled) * 0.8)
train, test = data_scaled[:train_size], data_scaled[train_size:]

# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Define sequence length
X_train, y_train = create_sequences(train, seq_length)
X_test, y_test = create_sequences(test, seq_length)

# Step 5: Model Definition
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(units=50, return_sequences=True),
    LSTM(units=50),
    Dense(units=1)
])

# Step 6: Model Compilation
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 7: Model Training
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Step 8: Model Evaluation
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

# Step 9: Forecasting
forecast_steps = 50
forecast_input = test[-seq_length:]  # Use the last sequence from test data for forecasting

forecast = []
for _ in range(forecast_steps):
    pred = model.predict(np.array([forecast_input]))
    forecast.append(pred[0, 0])
    forecast_input = np.append(forecast_input[1:], pred, axis=0)

# Inverse scaling for forecasted values
forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

print('Forecast:', forecast)
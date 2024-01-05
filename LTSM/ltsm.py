import pandas as pd
import numpy as np
# Load the CSV file
file_path = 'US_inflation_rates.csv'
data = pd.read_csv(file_path)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract the year from the 'date' column and create a new 'year' column
data['year'] = data['date'].dt.year

# Drop the original 'date' column if you no longer need it
# data.drop(columns=['date'], inplace=True)


# Convert the 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])

# Extract year and inflation rate
data['year'] = data['date'].dt.year
data = data[['year', 'value']]

# Set the 'year' column as the index
data.set_index('year', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Split data into train and test sets
train_size = int(len(data_scaled) * 0.8)
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # Length of sequences for LSTM
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build and train the LSTM model
model = Sequential()
model.add(LSTM(units=100, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=150, batch_size=16)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform predictions
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform(y_train)
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform(y_test)

# Create a DataFrame for plotting
plot_data = pd.DataFrame({
    'Year': np.concatenate((data.index[seq_length:seq_length+len(train_predict)],
                            data.index[seq_length+len(train_predict):seq_length+len(train_predict)+len(test_predict)])),
    'Predicted CPI': np.concatenate((train_predict.flatten(), test_predict.flatten())),
    'Actual CPI': np.concatenate((y_train.flatten(), y_test.flatten()))
})

# Create an interactive line plot using Plotly Express
fig = px.line(plot_data, x='Year', y=['Predicted CPI', 'Actual CPI'], title='CPI Forecasting with LSTM')
fig.update_layout(xaxis_title='Year', yaxis_title='CPI', legend_title='Data')
fig.show()

# For the training data
train_years = data.index[seq_length:len(y_train) + seq_length]
# For the testing data
test_years = data.index[len(y_train) + seq_length:len(y_train) + seq_length + len(y_test)]

# Combine years
combined_years = np.concatenate((train_years, test_years))

# Creating a DataFrame for the actual and predicted values
results = pd.DataFrame({
    'Year': combined_years,
    'Actual': np.concatenate((y_train.flatten(), y_test.flatten())),
    'Predicted': np.concatenate((train_predict.flatten(), test_predict.flatten()))
})

# Display the DataFrame
print(results.tail(13))  # Shows the first few rows

from sklearn.model_selection import ParameterGrid
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Define a parameter grid
param_grid = {
    'lstm_units': [20, 50, 100],
    'epochs': [50, 100, 150],
    'batch_size': [16, 32, 64]
}

# Initialize early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

best_mse = float('inf')
best_params = {}

# Iterate over all combinations of parameters
for params in ParameterGrid(param_grid):
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=params['lstm_units'], input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model with early stopping and validation split
    model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], callbacks=[early_stopping], validation_split=0.2)

    # Evaluate the model on the test set
    mse = model.evaluate(X_test, y_test)
    if mse < best_mse:
        best_mse = mse
        best_params = params

# Print the best Mean Squared Error and corresponding parameters
print(f"Best MSE: {best_mse}")
print(f"Best Params: {best_params}")

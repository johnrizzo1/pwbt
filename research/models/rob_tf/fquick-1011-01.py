import os
import argparse
from math import sqrt
from numpy import array
from pandas import DataFrame, Series, read_csv, to_datetime, concat
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from matplotlib import pyplot, colormaps
from datetime import datetime  # Import datetime to handle timestamps


# Logging function to handle log messages with timestamps
def logging(message):
    """Log messages with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current timestamp
    print(f"{timestamp} --> {message}", flush=True)  # Format the log message and flush the output immediately

# Convert time series to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

    n_vars = 1 if isinstance(data, list) else data.shape[1]

    df = DataFrame(data)

    cols, names = [], []

    

    logging("Creating input sequences.")

    # Input sequence (t-n ... t-1)

    for i in range(n_in, 0, -1):

        cols.append(df.shift(i))

        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]

    

    logging("Creating forecast sequences.")

    # Forecast sequence (t, t+1, ... t+n)

    for i in range(0, n_out):

        cols.append(df.shift(-i))

        if i == 0:

            names += [f'var{j+1}(t)' for j in range(n_vars)]

        else:

            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]

    

    agg = concat(cols, axis=1)

    agg.columns = names

    if dropnan:

        logging("Dropping rows with NaN values.")

        agg.dropna(inplace=True)

    return agg

# Create a differenced series
def difference(dataset, interval=1):

    logging("Creating differenced series.")

    return Series([dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))])

# Prepare data for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):

    logging("Preparing data for supervised learning.")

    raw_values = series.values

    diff_series = difference(raw_values, 1)

    diff_values = diff_series.values.reshape(len(diff_series), 1)

    

    scaler = MinMaxScaler(feature_range=(-1, 1))

    logging("Scaling values.")

    scaled_values = scaler.fit_transform(diff_values).reshape(len(diff_values), 1)

    

    supervised = series_to_supervised(scaled_values, n_lag, n_seq)

    supervised_values = supervised.values

    

    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]

    logging("Data preparation complete.")

    return scaler, train, test

# Forecast with LSTM model
def forecast_lstm(model, X, n_batch):
    logging("Forecasting with LSTM model.")
    X = X.reshape(1, 1, len(X))
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]

# Generate forecasts using the trained model and capture the corresponding timestamps
def make_forecasts_with_timestamps(model, n_batch, test, n_lag, timestamps):
    logging("Generating forecasts with timestamps.")
    forecasts = []
    for i in range(len(test)):
        X = test[i, :n_lag]
        forecast = forecast_lstm(model, X, n_batch)
        forecasts.append((timestamps[i], forecast))  # Store the timestamp and forecast
    return forecasts

# Invert differenced forecast to original scale
def inverse_difference(last_ob, forecast):
    logging("Inverting differenced forecast to original scale.")
    inverted = [forecast[0] + last_ob]
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# Inverse data transform to original scale
def inverse_transform(series, forecasts, scaler, offset):
    logging("Inverting transformed data to original scale.")
    inverted = []
    for i in range(len(forecasts)):
        forecast = array(forecasts[i]).reshape(1, len(forecasts[i]))
        inv_scale = scaler.inverse_transform(forecast)[0, :]
        last_ob = series.values[len(series) - offset + i - 1]
        inv_diff = inverse_difference(last_ob, inv_scale)
        inverted.append(inv_diff)  # Append the full array of inverted values
    return inverted

# Evaluate forecasts
def evaluate_forecasts(actual, forecasts, n_seq):
    logging("Evaluating forecasts.")
    for i in range(n_seq):
        actual_vals = [row[i] for row in actual]
        predicted_vals = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual_vals, predicted_vals))
        #logging(f'{i+1} RMSE: {rmse}')

# Plot forecasts
def plot_forecasts(series, forecasts, offset):
    logging("Plotting forecasts.")
    pyplot.plot(series.values, label='Original')
    cmap = colormaps.get_cmap('tab10')
    colors = [cmap(i / len(forecasts)) for i in range(len(forecasts))]

    for i, forecast in enumerate(forecasts):
        off_s = len(series) - offset + i - 1
        off_e = off_s + len(forecast) + 1
        xaxis = list(range(off_s, off_e))
        yaxis = [series.values[off_s]] + forecast

        pyplot.plot(xaxis, yaxis, color=colors[i], label=f'Forecast {i+1}')
        pyplot.scatter(xaxis, yaxis, color='black')

    pyplot.legend()
    pyplot.title('Forecast vs Actual')
    pyplot.xlabel('Time')
    pyplot.ylabel('Value')
    pyplot.show()

# Print forecast range
def print_forecast_range(forecasts):
    logging("Printing forecast range.")
    forecast_flat = [item for sublist in forecasts for item in sublist]
    forecast_min = min(forecast_flat)
    forecast_max = max(forecast_flat)
    logging(f"Predicted value range for the next ticks: Min = {forecast_min}, Max = {forecast_max}")

# Argument parser for model parameters
def get_parameters():
    logging("Getting model parameters from command line arguments.")
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lag', type=int, default=5, help='Number of lag observations (typically 5-10 for high-frequency trading)')
    parser.add_argument('--n_seq', type=int, default=5, help='Number of time steps to forecast (typically short-term, e.g., 5 for high-frequency trading)')
    parser.add_argument('--n_test', type=int, default=2, help='Number of test observations (larger for validation)')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs for training (adjust based on convergence)')
    parser.add_argument('--n_batch', type=int, default=1, help='Batch size for training (small for high-frequency trading)')
    parser.add_argument('--n_neurons', type=int, default=500, help='Number of neurons in LSTM layer (adjust based on data complexity)')
    parser.add_argument('--n_loops', type=int, default=1000, help='Number of times to run the main loop')  # Added n_loops argument
    return parser.parse_args()

# Main function
def main():
    params = get_parameters()
    for _ in range(params.n_loops):  # Loop through main process based on n_loops parameter
        logging(f"Starting loop {_ + 1}/{params.n_loops}")

        # Load dataset
        try:
            logging("Loading dataset.")
            #series = read_csv('C:/TEMP/db212.csv', header=0, parse_dates=[0], index_col=0, dtype=str)
            series = read_csv('C:/Log/RGW.csv', header=0, parse_dates=[0], index_col=0, dtype=str)
            series.index = to_datetime(series.index, errors='coerce')  # Use errors='coerce' to handle any inconsistent date formats
            series = series.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric, coercing errors
            series = series.dropna()  # Drop rows with NaN values after conversion
            logging("Dataset loaded successfully.")
        except FileNotFoundError:
            logging("Dataset not found!")
            return
        except Exception as e:
            logging(f"Error loading dataset: {e}")
            return

        # Prepare data
        scaler, train, test = prepare_data(series, params.n_test, params.n_lag, params.n_seq)

        # Get the corresponding timestamps for the test data
        test_timestamps = series.index[-params.n_test:]

        # Train or load model
        model_filename = 'C:/TEMP/lstm_model.h5'
        logging("Loading model...")
        model = load_model(model_filename)

        # Generate forecasts with timestamps
        forecasts_with_timestamps = make_forecasts_with_timestamps(model, params.n_batch, test, params.n_lag, test_timestamps)
        forecasts = [forecast for _, forecast in forecasts_with_timestamps]

        # Inverse transform to original scale
        forecasted_actuals = inverse_transform(series, forecasts, scaler, params.n_test - 1)

        # Evaluate and plot
        actual = inverse_transform(series, [row[params.n_lag:] for row in test], scaler, params.n_test - 1)
        evaluate_forecasts(actual, forecasted_actuals, params.n_seq)

        # Print all predicted values with their corresponding timestamps and forecasted actual values
        logging("Predicted values with their corresponding timestamps:")
        for (timestamp, forecast), forecasted_actual in zip(forecasts_with_timestamps, forecasted_actuals):
            logging("-" * 40)
            last_row_value = series.iloc[-1].values[0]  # Access the actual value from the last row
            logging(f"Last row of the series: {last_row_value}")
            logging(f"Forecasted actual values: {forecasted_actual}")
            logging("-" * 40)

        # Print forecast range
        print_forecast_range(forecasts)

# Run the program
if __name__ == '__main__':
    main()
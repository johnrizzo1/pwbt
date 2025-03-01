import os
import shutil
import argparse
from numpy import array
import pandas as pd
from pandas import DataFrame, Series, read_csv, to_datetime, concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from datetime import datetime

# Logging function to handle log messages with timestamps
def logging(message):
    """Log messages with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} --> {message}", flush=True)

# Convert time series to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """Transform time series data into a supervised learning format."""
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = DataFrame(data)
    cols, names = [], []

    # Input sequence (t-n ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
    logging("Created input sequences for supervised learning.")

    # Forecast sequence (t, t+1, ... t+n)
    for i in range(n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [f'var{j+1}(t)' for j in range(n_vars)]
        else:
            names += [f'var{j+1}(t+{i})' for j in range(n_vars)]
    logging("Created forecast sequences for supervised learning.")

    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)  # Remove rows with NaN values
    logging("Aggregated input and forecast sequences into a single DataFrame.")
    return agg

# Create a differenced series
def difference(dataset, interval=1):
    """Create a differenced series."""
    result = Series([dataset[i] - dataset[i - interval] for i in range(interval, len(dataset))])
    logging("Created a differenced series.")
    return result

# Prepare data for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    """Prepare the time series data for supervised learning."""
    raw_values = series.values
    diff_series = difference(raw_values)
    diff_values = diff_series.values.reshape(len(diff_series), 1)

    # Scale the data to the range [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values).reshape(len(diff_values), 1)
    logging("Scaled the differenced data to the range [-1, 1].")

    # Convert scaled values to supervised learning format
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values

    # Split into training and testing sets
    train, test = supervised_values[:-n_test], supervised_values[-n_test:]
    logging("Prepared training and testing sets.")

    return scaler, train, test

# Fit LSTM model
def fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons):
    """Fit an LSTM model to the training data."""
    X, y = train[:, :n_lag], train[:, n_lag:]  # Split input and output
    X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM
    logging("Reshaped input data for LSTM.")

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]),
                   stateful=True, return_sequences=True, dropout=0.2))
    model.add(LSTM(n_neurons, stateful=True, dropout=0.2))
    model.add(Dense(y.shape[1]))  # Output layer
    model.compile(loss='mean_squared_error', optimizer='adam')
    logging("LSTM model built and compiled.")

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

    # Fit the model
    for epoch in range(n_epochs):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False, callbacks=[early_stopping])
        model.reset_states()  # Reset states after each epoch
        logging(f"Epoch {epoch + 1}/{n_epochs} completed.")

    return model

# Argument parser for model parameters
def get_parameters():
    """Parse command line parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lag', type=int, default=5, help='Number of lag observations (5-10 for high-frequency trading)')
    parser.add_argument('--n_seq', type=int, default=5, help='Number of time steps to forecast (short-term)')
    parser.add_argument('--n_test', type=int, default=2, help='Number of test observations (larger for validation)')
    parser.add_argument('--n_epochs', type=int, default=2, help='Number of epochs for training (adjust for convergence)')
    parser.add_argument('--n_batch', type=int, default=1, help='Batch size for training (small for high-frequency trading)')
    parser.add_argument('--n_neurons', type=int, default=500, help='Number of neurons in LSTM layer (adjust based on data complexity)')
    parser.add_argument('--n_looping', type=int, default=1000, help='Number of loops to run the main logic')
    logging("Parsed command line parameters.")
    return parser.parse_args()

# File backup function
def backup_model(model_filename):
    """Backup the model file by appending '_backup'."""
    # Extract the directory and base name of the file
    directory, filename = os.path.split(model_filename)
    basename, ext = os.path.splitext(filename)

    # Create a backup filename that is called *active
    backup_filename = os.path.join(directory, f"{basename}_active{ext}")

    # Copy the model file to the backup location
    try:
        shutil.copy(model_filename, backup_filename)
        logging(f"Backup created successfully: {backup_filename}")
    except Exception as e:
        logging(f"Error creating backup: {e}")

# Main function
def main():
    """Main function to execute the LSTM model training."""
    logging("Starting the program.")  # Log that the program is starting
    params = get_parameters()

    for loop in range(params.n_looping):
        logging(f"Starting loop {loop + 1}/{params.n_looping}")
        # Load the dataset
        try:
            file_path = r'G:\My Drive\rgw.co\WiZer\datasciai.com\Log\1011\v1.1\1011_price.csv'
            if os.path.exists(file_path):
                series = read_csv(file_path, header=0, parse_dates=[0], index_col=0, dtype=str)
                series = series.tail(100)
                series.index = to_datetime(series.index, errors='coerce')
                series = series.apply(pd.to_numeric, errors='coerce')
                series = series.dropna()
                logging("Dataset loaded successfully.")
            else:
                logging(f"Dataset not found at: {file_path}")
        except Exception as e:
            logging(f"Error loading dataset: {e}")
            return

        # Prepare data for training and testing
        scaler, train, test = prepare_data(series, params.n_test, params.n_lag, params.n_seq)

        # Define the model filename
        model_filename = r'G:/My Drive/rgw.co/WiZer/datasciai.com/Log/1011/v1.1/1011_lstm_model.h5'

        # Backup the model
        backup_model(model_filename)

        # Train the LSTM model
        logging("Training model...")
        model = fit_lstm(train, params.n_lag, params.n_seq, params.n_batch, params.n_epochs, params.n_neurons)
        model.save(model_filename)  # Save the trained model
        logging(f"Model training complete and saved for loop {loop + 1}/{params.n_looping}.")


# Run the program
if __name__ == '__main__':
    main()
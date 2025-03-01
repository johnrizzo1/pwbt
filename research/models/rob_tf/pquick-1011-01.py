import time
import argparse
import os
import csv
from datetime import datetime
from io import StringIO
from trade29.sc.bridge import SCBridge
import pandas as pd
import numpy as np
import json
import re

# This will be the global memory structure that stores the main log data
memory_structure = []
new_rows_count = 0  # Counter for the number of new rows processed

# Default directory and file setup
DEFAULT_DIR = r"C:\Log"
KEY = "RGW"  # Set this as required (or passed as a parameter)
OUTPUT_FILE = os.path.join(DEFAULT_DIR, f"{KEY}.csv")
OUTPUT_FILE = r'G:\My Drive\rgw.co\WiZer\datasciai.com\Log\1011\v1.1\1011_price.csv'

# Define the main logging function for data to be written to file
def main_data_logging(timestamp: str, message: str, output_file):
    log_entry = f"{timestamp}, {message}"  # Create the log entry
    print(log_entry)  # Print the timestamped message with a comma separator
    memory_structure.append(log_entry)  # Store the log entry in memory

    # Write to file after 60 rows are logged
    global new_rows_count
    new_rows_count += 1

    # Check if 60 new rows have been processed
    if new_rows_count >= 1:
        write_to_file(output_file)  # Write the data to the file
        new_rows_count = 0  # Reset the row count after writing to the file

# Define the secondary logging function for non-data messages (comments, system messages, etc.)
def comment_logging(message: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{timestamp} - {message}")  # Just print the comment to the console

# Function to write logs to the CSV file after 60 new rows are processed
def write_to_file(output_file):
    # Make sure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write data to CSV file
    file_exists = os.path.exists(output_file)
    with open(output_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write headers if the file is new
            writer.writerow(['Timestamp', 'Data'])
        # Write the data
        for entry in memory_structure:
            timestamp, data = entry.split(", ", 1)
            writer.writerow([timestamp, data])

    # Clear memory after writing
    memory_structure.clear()

# Function to initialize the SCBridge instance and start the data request process
def initialize_scbridge():
    comment_logging("Starting SCBridge data request process...")

    # Create an instance of the SCBridge class
    comment_logging("Creating SCBridge instance...")
    sc = SCBridge()
    comment_logging("SCBridge instance created.")
    
    # Request data from SCBridge
    comment_logging("Requesting data from SCBridge...")
    sc.graph_data_request(
        key="RGW",  # ES key
        base_data='4',  # Open, High, Low, Close
        include_timestamp='0',
        update_frequency='1000',
    )
    comment_logging("Data request sent to SCBridge.")
    
    return sc

def process_data(sc, output_file):
    """
    Continuously receives and processes data from the SCBridge.
    
    Parameters:
    - sc: The SCBridge instance to get data from.
    """
    comment_logging("Entering loop to receive and process data...")
    
    while True:
        try:
            # Wait for data from SCBridge with a timeout to avoid indefinite blocking
            msg = sc.get_response_queue().get(timeout=10)
            
            if msg:
                # Iterate over each row of the DataFrame and log the specific value (e.g., Close)
                for index, row in msg.df.iterrows():
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    main_data_logging(timestamp, row[1], output_file)  # Assuming 'Close' is at index 1

        except Exception as e:
            comment_logging(f"Connection error or timeout occurred: {e}")
            break  # Exit the loop if an error occurs

    comment_logging("Exiting program.")

def main():
    """
    Main procedure to initialize the SCBridge and process data.
    """
    # Argument parsing to allow passing of output file path
    parser = argparse.ArgumentParser(description="Write data to file after 60 rows.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output file path (default is C:\\Log\\KEY.csv).")
    args = parser.parse_args()

    # Initialize SCBridge and start data request
    sc = initialize_scbridge()

    # Process the data continuously
    process_data(sc, args.output)

# For direct execution (optional)
if __name__ == "__main__":
    main()

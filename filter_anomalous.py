import pandas as pd
import os
import sys
import csv

def filter_anomalous_data(input_file, output_file=None, column_name='anomalous', value=1):
    """
    Filter rows where specified column equals specified value and save to a new CSV file.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str): Path to the output CSV file. If None, will auto-generate name
    column_name (str): Name of column to filter on (default: 'anomalous')
    value: Value to filter for (default: 1)
    """
    # Generate output filename if not provided
    if output_file is None:
        # If input filename contains "anomalous", replace it with "filtered"
        if "anomalous" in input_file:
            # Replace "anomalous" with "filtered" in both directory and filename
            output_file = input_file.replace("anomalous", "filtered")
            
            # Make sure the directory exists
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created directory: {output_dir}")
        else:
            # Otherwise use the standard pattern
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_filtered.csv"
    
    try:
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return
            
        # Read the CSV file with automatic delimiter detection
        print(f"Reading data from {input_file}...")
        
        # Try to detect delimiter automatically
        with open(input_file, 'r') as f:
            first_line = f.readline().strip()
            dialect = csv.Sniffer().sniff(first_line)
            delimiter = dialect.delimiter
        
        print(f"Detected delimiter: '{delimiter}'")
        df = pd.read_csv(input_file, sep=delimiter)
        
        # Check if specified column exists
        if column_name not in df.columns:
            print(f"Error: '{column_name}' column not found in the CSV file.")
            print("Available columns are:")
            for col in df.columns:
                print(f"  - {col}")
            return
        
        # Filter rows where column = value
        filtered_df = df[df[column_name] == value]
        
        # Save filtered data to new CSV file using the same delimiter as the input
        filtered_df.to_csv(output_file, index=False, sep=delimiter)
        
        print(f"Filtered {len(filtered_df)} records out of {len(df)} total records.")
        print(f"Saved filtered data to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Trying with common delimiters...")
        
        # Try common delimiters if automatic detection fails
        for sep in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(input_file, sep=sep)
                if column_name in df.columns:
                    print(f"Successfully read file using delimiter: '{sep}'")
                    filtered_df = df[df[column_name] == value]
                    
                    filtered_df.to_csv(output_file, index=False, sep=sep)
                    print(f"Filtered {len(filtered_df)} records out of {len(df)} total records.")
                    print(f"Saved filtered data to {output_file}")
                    return
            except:
                continue
        
        print("Failed to read the file with any common delimiter.")

if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python filter_anomalous.py input1.csv [input2.csv input3.csv ...] [column_name]")
        print("Default column_name is 'anomalous' if not specified")
        sys.exit(1)
    
    # The last argument might be the column name if it doesn't end with .csv
    args = sys.argv[1:]
    column_name = 'anomalous'
    value = 1
    
    if not args[-1].lower().endswith('.csv'):
        column_name = args[-1]
        input_files = args[:-1]
        
        # Check if there's a value specification after column name
        if len(sys.argv) > len(input_files) + 2 and sys.argv[len(input_files) + 2].isdigit():
            value = int(sys.argv[len(input_files) + 2])
    else:
        input_files = args
    
    # Process each input file
    for input_file in input_files:
        print(f"\nProcessing {input_file}...")
        filter_anomalous_data(input_file, None, column_name, value)
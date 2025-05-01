import sys
import pandas as pd
import os

# Check if at least one filename is provided
if len(sys.argv) < 2:
    print("Usage: python file.py <file1.csv> <file2.csv> ...")
    sys.exit(1)

# Loop through each file passed as argument
for filename in sys.argv[1:]:
    try:
        df = pd.read_csv(filename)

        # Add anomalous column
        df['anomalous'] = (
            (df['voltage'] < 217.4) | (df['voltage'] > 242.6) |
            (df['frequency'] < 59.2) | (df['frequency'] > 60.8) |
            (df['powerFactor'] < 0.792) | (df['powerFactor'] > 1.0) |
            (df['current'] < 0) | (df['current'] > 50) |
            (df['power'] < 0) | (df['power'] > 10000)
        ).astype(int)

        # Build new filename
        base_name = os.path.splitext(filename)[0]
        save_file = base_name + '_anomalous.csv'

        # Save updated file
        df.to_csv(save_file, index=False)
        print(f"[✓] Saved: {save_file}")

    except Exception as e:
        print(f"[✗] Failed to process '{filename}': {e}")
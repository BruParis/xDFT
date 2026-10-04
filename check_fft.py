import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def compute_and_display_fft(csv_file):
    # Load CSV file
    data = pd.read_csv(csv_file)

    # Extract real and imaginary parts
    if "real" not in data.columns or "imaginary" not in data.columns:
        raise ValueError("CSV must have 'real' and 'imaginary' columns")

    real = data["real"].values
    imaginary = data["imaginary"].values

    # Combine real and imaginary parts into a complex signal
    signal = real + 1j * imaginary
    print("signal:")
    for row in signal:
        print(row)

    # Compute FFT
    fft_result = np.fft.fft(signal, norm="forward")

    print("FFT result:")
    for row in fft_result:
        print(row)


# Example usage
csv_file = "input.csv"  # Replace with your CSV file path
compute_and_display_fft(csv_file)

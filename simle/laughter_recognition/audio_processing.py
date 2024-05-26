import os
import pandas as pd
import librosa
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.python import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.utils import to_categorical

from keras.layers import Input

# Analyze the main frequency component functions within a given frequency range in the audio file
def analyze_frequency_range(audio_file, min_freq, max_freq):
    """
     Analyzes the main frequency components within a given frequency range in an audio file

     parameter:
     - audio_file: audio file path
     - min_freq: minimum frequency
     - max_freq: maximum frequency

     return value:
     - main_frequencies: main frequency components within a given frequency range
     """
    y, sr = librosa.load(audio_file)
    # Calculate spectrum
    D = np.abs(librosa.stft(y))
    # Get the frequency index within the frequency range
    freqs = librosa.fft_frequencies(sr=sr)
    freq_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))
    # Get the main frequency components within a given frequency range
    main_frequencies = np.argmax(D[freq_idx], axis=0)
    return main_frequencies

# Calculate the energy distribution function within a given frequency range in the audio file
def calculate_energy_distribution(audio_file, min_freq, max_freq):
"""
     Calculate the energy distribution within a given frequency range in an audio file

     parameter:
     - audio_file: audio file path
     - min_freq: minimum frequency
     - max_freq: maximum frequency

     return value:
     - energy_distribution: energy distribution within a given frequency range
     """
    y, sr = librosa.load(audio_file)
    # Calculate spectrum
    D = np.abs(librosa.stft(y))
    # Get the frequency index within the frequency range
    freqs = librosa.fft_frequencies(sr=sr)
    freq_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))
    # Calculate energy distribution
    energy_distribution = np.sum(D[freq_idx], axis=0)
    return energy_distribution

# Extract the time domain feature function of the audio file
def extract_time_domain_features(audio_file):
"""
     Extract time domain features of audio files

     parameter:
     - audio_file: audio file path

     return value:
     - time_domain_features: Dictionary of time domain features
     """
    y, sr = librosa.load(audio_file)
    time_domain_features = {
        'Peak Amplitude': np.max(y),
        'Root Mean Square (RMS)': np.sqrt(np.mean(y**2)),
        'Zero Crossing Rate (ZCR)': librosa.feature.zero_crossing_rate(y)[0, 0]
    }
    return time_domain_features

# Laughter threshold judgment
def is_laughter(features):
    # Get the required information from the features
    main_frequencies = features['Main Frequencies']
    energy_distribution = features['Energy Distribution']
    time_domain_features = features['Time Domain Features']

    # Define laughter judgment rules
     # Example rule: If the dominant frequency is contained within a specific range, and the energy distribution is above a certain threshold, and the time domain characteristics meet certain conditions, it is considered laughter
    if (100 in main_frequencies) and (500 in main_frequencies):
        if np.max(energy_distribution) > 1000:  # Set energy threshold
            if time_domain_features['Zero Crossing Rate (ZCR)'] > 0.5:  #Set time domain feature threshold
                return True

    return False

# Traverse the subfolders of the training data set and test data set
def process_data(data_folder):
    data = []
    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if file.endswith('.wav'):
                   # 1. Analyze frequency range, calculate energy distribution, and extract time domain features
                    main_frequencies = analyze_frequency_range(file_path, 100, 500)
                    energy_distribution = calculate_energy_distribution(file_path, 100, 500)
                    time_domain_features = extract_time_domain_features(file_path)
                    # 2. Judgment of laughter threshold
                    laughter_detected = is_laughter({'Main Frequencies': main_frequencies,
                                                     'Energy Distribution': energy_distribution,
                                                     'Time Domain Features': time_domain_features})
                    #tagged for laughter
                    laughter_detected = True  # All audio data is laughter
                    # adding data
                    data.append([file, {'Main Frequencies': main_frequencies,
                                        'Energy Distribution': energy_distribution,
                                        'Time Domain Features': time_domain_features},
                                 laughter_detected])

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['File', 'Features', 'Laughter Detected'])
    return df

# Read the training data set and test data set
train_data_folder = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train"
test_data_folder = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test"

# Data processing
train_df = process_data(train_data_folder)
test_df = process_data(test_data_folder)

print(train_df.head())
print(test_df.head())

train_df.to_csv("/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train_processed.csv", index=False)
test_df.to_csv("/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test_processed.csv", index=False)

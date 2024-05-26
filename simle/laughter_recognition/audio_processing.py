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

# 分析音频文件中给定频率范围内的主要频率成分函数
def analyze_frequency_range(audio_file, min_freq, max_freq):
    """
    分析音频文件中给定频率范围内的主要频率成分

    参数：
    - audio_file: 音频文件路径
    - min_freq: 最小频率
    - max_freq: 最大频率

    返回值：
    - main_frequencies: 给定频率范围内的主要频率成分
    """
    y, sr = librosa.load(audio_file)
    # 计算频谱
    D = np.abs(librosa.stft(y))
    # 获取频率范围内的频率索引
    freqs = librosa.fft_frequencies(sr=sr)
    freq_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))
    # 获取给定频率范围内的主要频率成分
    main_frequencies = np.argmax(D[freq_idx], axis=0)
    return main_frequencies

# 计算音频文件中给定频率范围内的能量分布函数
def calculate_energy_distribution(audio_file, min_freq, max_freq):
    """
    计算音频文件中给定频率范围内的能量分布

    参数：
    - audio_file: 音频文件路径
    - min_freq: 最小频率
    - max_freq: 最大频率

    返回值：
    - energy_distribution: 给定频率范围内的能量分布
    """
    y, sr = librosa.load(audio_file)
    # 计算频谱
    D = np.abs(librosa.stft(y))
    # 获取频率范围内的频率索引
    freqs = librosa.fft_frequencies(sr=sr)
    freq_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))
    # 计算能量分布
    energy_distribution = np.sum(D[freq_idx], axis=0)
    return energy_distribution

# 提取音频文件的时域特征函数
def extract_time_domain_features(audio_file):
    """
    提取音频文件的时域特征

    参数：
    - audio_file: 音频文件路径

    返回值：
    - time_domain_features: 时域特征字典
    """
    y, sr = librosa.load(audio_file)
    time_domain_features = {
        'Peak Amplitude': np.max(y),
        'Root Mean Square (RMS)': np.sqrt(np.mean(y**2)),
        'Zero Crossing Rate (ZCR)': librosa.feature.zero_crossing_rate(y)[0, 0]
    }
    return time_domain_features

# 笑声阈值判断
def is_laughter(features):
    # 从特征中获取所需的信息
    main_frequencies = features['Main Frequencies']
    energy_distribution = features['Energy Distribution']
    time_domain_features = features['Time Domain Features']

    # 定义笑声判断规则
    # 示例规则：如果主要频率包含在特定范围内，且能量分布在某个阈值以上，并且时域特征满足某些条件，则认为是笑声
    if (100 in main_frequencies) and (500 in main_frequencies):
        if np.max(energy_distribution) > 1000:  # 设置能量阈值
            if time_domain_features['Zero Crossing Rate (ZCR)'] > 0.5:  # 设置时域特征阈值
                return True

    return False

# 遍历训练数据集和测试数据集的子文件夹
def process_data(data_folder):
    data = []
    for subdir in os.listdir(data_folder):
        subdir_path = os.path.join(data_folder, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file)
                if file.endswith('.wav'):
                    # 1. 分析频率范围，计算能量分布，提取时域特征
                    main_frequencies = analyze_frequency_range(file_path, 100, 500)
                    energy_distribution = calculate_energy_distribution(file_path, 100, 500)
                    time_domain_features = extract_time_domain_features(file_path)
                    # 2. 笑声阈值判断
                    laughter_detected = is_laughter({'Main Frequencies': main_frequencies,
                                                     'Energy Distribution': energy_distribution,
                                                     'Time Domain Features': time_domain_features})
                    # 标记为笑声
                    laughter_detected = True  # 所有音频数据均为笑声
                    # 添加数据
                    data.append([file, {'Main Frequencies': main_frequencies,
                                        'Energy Distribution': energy_distribution,
                                        'Time Domain Features': time_domain_features},
                                 laughter_detected])

    # 转换为 DataFrame
    df = pd.DataFrame(data, columns=['File', 'Features', 'Laughter Detected'])
    return df

# 读取训练数据集和测试数据集
train_data_folder = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train"
test_data_folder = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test"

# 处理数据
train_df = process_data(train_data_folder)
test_df = process_data(test_data_folder)

# 输出处理后的数据
print(train_df.head())
print(test_df.head())

# 保存处理后的数据到CSV文件
train_df.to_csv("/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train_processed.csv", index=False)
test_df.to_csv("/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test_processed.csv", index=False)

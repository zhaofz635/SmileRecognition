import cv2
import csv
import numpy as np
import librosa
from moviepy.editor import VideoFileClip
from tensorflow.keras.models import load_model
import os

# 提取视频和音频特征
def extract_video_audio_features(video_file, segment_length=1):
    frames = []
    face_positions = []
    smile_ratios = []
    audio_segments = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    video_clip = VideoFileClip(video_file)
    fps = video_clip.fps
    num_frames_per_segment = int(fps * segment_length)

    # 提取音频
    audio_path = "temp_audio.wav"
    video_clip.audio.write_audiofile(audio_path)

    # 处理视频帧
    for i, frame in enumerate(video_clip.iter_frames()):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        current_smile_ratios = []

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            smile_ratio = len(smiles) / (w * h)
            current_smile_ratios.append(smile_ratio)

        if len(faces) > 0:
            face_positions.append([x, y, w, h])
            smile_ratios.append(current_smile_ratios[0] if current_smile_ratios else 0)
        else:
            face_positions.append([0, 0, 0, 0])
            smile_ratios.append(0)

        frames.append(gray_frame)

        # 每段处理完毕，保存到segments
        if (i + 1) % num_frames_per_segment == 0:
            audio_segment = extract_audio_segment(audio_path, i // fps, segment_length)
            audio_segments.append(audio_segment)
            yield frames, face_positions, smile_ratios, audio_segment
            frames, face_positions, smile_ratios = [], [], []

    video_clip.close()

    # 删除临时音频文件
    if os.path.exists(audio_path):
        os.remove(audio_path)

def extract_audio_segment(audio_path, start_time, duration):
    y, sr = librosa.load(audio_path, offset=start_time, duration=duration)
    main_frequencies = analyze_frequency_range(y, sr, 100, 500)
    energy_distribution = calculate_energy_distribution(y, sr, 100, 500)
    time_domain_features = extract_time_domain_features(y, sr)
    audio_features = list(main_frequencies) + list(energy_distribution) + list(time_domain_features)
    return audio_features

def analyze_frequency_range(y, sr, min_freq, max_freq):
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    freq_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))
    main_frequencies = np.argmax(D[freq_idx], axis=0)
    return main_frequencies

def calculate_energy_distribution(y, sr, min_freq, max_freq):
    D = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    freq_idx = np.where((freqs >= min_freq) & (freqs <= max_freq))
    energy_distribution = np.sum(D[freq_idx], axis=0)
    return energy_distribution

def extract_time_domain_features(y, sr):
    peak_amplitude = np.max(y)
    rms = np.sqrt(np.mean(y ** 2))
    zcr_result = librosa.feature.zero_crossing_rate(y)
    zcr = np.mean(zcr_result[0])
    time_domain_features = [peak_amplitude, rms, zcr]
    return time_domain_features

def preprocess_audio_data(data, target_length=2039):
    data = np.array(data)
    print("Original audio data length:", len(data))
    if len(data) < target_length:
        pad_width = (target_length - len(data),)
        data = np.pad(data, (0, pad_width[0]), 'constant')
    elif len(data) > target_length:
        data = data[:target_length]
    data = np.expand_dims(data, axis=0)
    data = np.expand_dims(data, axis=-1)
    print("Preprocessed audio data shape:", data.shape)
    return data

def preprocess_face_data(face_positions, smile_ratios, target_length=2039):
    face_positions = np.array(face_positions).flatten()
    smile_ratios = np.array(smile_ratios).flatten()
    combined_features = np.concatenate((face_positions, smile_ratios))
    print("Combined face data length:", len(combined_features))

    if len(combined_features) < target_length:
        pad_width = (target_length - len(combined_features),)
        combined_features = np.pad(combined_features, (0, pad_width[0]), 'constant')
    elif len(combined_features) > target_length:
        combined_features = combined_features[:target_length]

    combined_features = np.expand_dims(combined_features, axis=0)
    combined_features = np.expand_dims(combined_features, axis=-1)
    print("Preprocessed face data shape:", combined_features.shape)
    return combined_features

# 加载模型
model_path = '/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/laughter_detection_model.h5'  # 更新为你的模型路径
model = load_model(model_path, compile=False)

def generate_feedback(face_pred, laugh_pred):
    if face_pred and laugh_pred:
        return "Laughter and Smiley detected"
    elif laugh_pred:
        return "Laughter detected"
    elif face_pred:
        return "Smiley detected"
    else:
        return "No smiley or laughter detected"

def save_to_csv(output_file, data):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV文件的表头
        writer.writerow(['FileName', 'LaughingBehavior', 'BehaviorType', 'StartingTime', 'EndingTime'])
        # 写入笑行为数据
        if data:
            for row in data:
                writer.writerow(row)
        else:
            writer.writerow(['None', 'None Behavior detected', 'None', 'None', 'None'])

def recognize_laughter(video_audio_file):
    segment_length = 1  # 每个片段的持续时间（秒）
    laughter_intervals = []
    laughter_data = []  # 用于存储笑行为数据

    for i, (frames, face_positions, smile_ratios, audio_features) in enumerate(extract_video_audio_features(video_audio_file, segment_length)):
        preprocessed_audio_features = preprocess_audio_data(audio_features, 2039)
        preprocessed_face_features = preprocess_face_data(face_positions, smile_ratios, 2039)

        predictions = model.predict([preprocessed_audio_features, preprocessed_face_features])
        print("Predictions:", predictions)

        # 检查预测输出的形状
        if len(predictions.shape) == 2:
            laughter_probabilities = predictions[0]
        else:
            raise ValueError("Unexpected model output shape.")

        print("Laughter probabilities:", laughter_probabilities)

        # 根据阈值判断笑声和笑脸的检测结果
        face_pred = laughter_probabilities[0] > 0.5
        laugh_pred = laughter_probabilities[1] > 0.5

        feedback = generate_feedback(face_pred, laugh_pred)
        print("Feedback:", feedback)

        # 保存检测到的笑脸和笑声的时间段
        if laugh_pred or face_pred:
            laughter_intervals.append((i * segment_length, (i + 1) * segment_length))
            laughter_data.append([
                video_audio_file,
                feedback,
                'laughter' if laugh_pred else 'smiley' if face_pred else 'none',
                f"{i * segment_length}",
                f"{(i + 1) * segment_length}"
            ])

    save_to_csv('laughter_data.csv', laughter_data)

    return laughter_intervals

# 调用recognize_laughter函数进行测试
video_audio_file_path = '/Users/fuzhengzhao/Desktop/test.mp4'  # 更新为实际的视频文件路径
laughter_intervals = recognize_laughter(video_audio_file_path)
print("Laughter intervals:", laughter_intervals)

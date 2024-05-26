import os
import pandas as pd
import numpy as np
import audio_processing
from keras.models import Model, save_model
from keras.layers import Input, Conv1D, MaxPooling1D, Dense, Dropout, concatenate, GlobalMaxPooling1D, Reshape, LSTM
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# define path
train_data_folder = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train"
test_data_folder = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test"
face_data_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/output.csv"

# Data audio preparation and preprocessing
train_df = audio_processing.process_data(train_data_folder)
test_df = audio_processing.process_data(test_data_folder)

#Extract audio features
max_feature_length = max(len(features['Main Frequencies']) + len(features['Energy Distribution']) + len(features['Time Domain Features']) for features in train_df['Features'])

def extract_features(df, max_len):
    return np.array([
        np.concatenate((
            features['Main Frequencies'],
            features['Energy Distribution'],
            list(features['Time Domain Features'].values()),
            [0] * (max_len - len(features['Main Frequencies']) - len(features['Energy Distribution']) - len(features['Time Domain Features']))
        )) for features in df['Features']
    ])

X_train_audio = extract_features(train_df, max_feature_length)
X_test_audio = extract_features(test_df, max_feature_length)

y_train_audio = to_categorical(train_df['Laughter Detected'], num_classes=2)
y_test_audio = to_categorical(test_df['Laughter Detected'], num_classes=2)

# Read the data characteristics and labels of the smiley face
face_df = pd.read_csv(face_data_path, index_col=False)

# Check if the 'Face Position' and 'Smile Ratio' columns exist
if 'Face Position' not in face_df.columns or 'Smile Ratio' not in face_df.columns:
    raise KeyError("'Face Position' or 'Smile Ratio' column is missing from the DataFrame")

# Filter out invalid values
face_df = face_df.dropna(subset=['Face Position', 'Smile Ratio'])

# Convert the 'Face Position' column to numeric data
def convert_face_position(x):
    try:
        return tuple(map(int, x.strip('()').split(',')))
    except Exception as e:
        return None

face_df['Face Position'] = face_df['Face Position'].apply(convert_face_position)
# Delete rows that failed to convert
face_df = face_df.dropna(subset=['Face Position'])  

# Delete the 'Video Number' column
face_df = face_df.drop(columns=['Video Number'])

# Split the face dataset based on the number of audio datasets
num_train_samples = len(train_df)
num_test_samples = len(test_df)

face_train_df = face_df.iloc[:num_train_samples]
face_test_df = face_df.iloc[num_train_samples:num_train_samples + num_test_samples]

# Debugging: Print shapes to ensure consistency
print("Shape of X_train_audio:", X_train_audio.shape)
print("Shape of X_test_audio:", X_test_audio.shape)
print("Number of training samples for face data:", len(face_train_df))
print("Number of testing samples for face data:", len(face_test_df))

# Get the characteristic data of the smiley face
face_features_train = face_train_df.apply(lambda row: list(row['Face Position']) + [row['Smile Ratio']], axis=1).tolist()
face_features_test = face_test_df.apply(lambda row: list(row['Face Position']) + [row['Smile Ratio']], axis=1).tolist()

# Debugging: Check if face features have the same length as audio samples
print("Number of face features for training:", len(face_features_train))
print("Number of face features for testing:", len(face_features_test))

# Get the label data of the smiley face
y_train_face = to_categorical(face_train_df['Detected'], num_classes=2)
y_test_face = to_categorical(face_test_df['Detected'], num_classes=2)

# Debugging: Check label shapes
print("Shape of y_train_face:", y_train_face.shape)
print("Shape of y_test_face:", y_test_face.shape)

# Merge the data features of the smiley face into the data features of the audio
X_train_face = np.array(face_features_train, dtype=np.float32)
X_test_face = np.array(face_features_test, dtype=np.float32)

# Determine the maximum length of facial features
max_face_feature_length = max([len(features) for features in face_features_train])

# Correct the length of the smile feature to match the length of the audio feature
X_train_face = np.pad(X_train_face, ((0, 0), (0, max_feature_length - max_face_feature_length)), 'constant')
X_test_face = np.pad(X_test_face, ((0, 0), (0, max_feature_length - max_face_feature_length)), 'constant')

# Reshape features into a shape suitable for the CNN input
X_train_audio = np.expand_dims(X_train_audio, axis=2)
X_test_audio = np.expand_dims(X_test_audio, axis=2)
X_train_face = np.expand_dims(X_train_face, axis=2)
X_test_face = np.expand_dims(X_test_face, axis=2)

# Debugging: Check final shapes before training and evaluation
print("Final shape of X_train_audio:", X_train_audio.shape)
print("Final shape of X_test_audio:", X_test_audio.shape)
print("Final shape of X_train_face:", X_train_face.shape)
print("Final shape of X_test_face:", X_test_face.shape)

# Confirm the shape of the training and test set data
if X_train_audio.shape[0] != X_train_face.shape[0]:
    raise ValueError("Mismatch between number of audio and face data samples in training set")

if X_test_audio.shape[0] != X_test_face.shape[0]:
    raise ValueError("Mismatch between number of audio and face data samples in test set")

# Build CNN models for audio and smiley faces and fuse them
def create_multimodal_model(input_shape_audio, input_shape_face):
    # Audio feature extraction
    audio_input = Input(shape=input_shape_audio)
    x = Conv1D(filters=64, kernel_size=2, activation='relu')(audio_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    audio_output = Reshape((128,))(x)

    # Face feature extraction
    face_input = Input(shape=input_shape_face)
    y = Conv1D(filters=64, kernel_size=2, activation='relu')(face_input)
    y = MaxPooling1D(pool_size=2)(y)
    y = LSTM(64, return_sequences=True)(y)
    y = GlobalMaxPooling1D()(y)
    y = Dense(128, activation='relu')(y)
    y = Dropout(0.5)(y)
    face_output = Reshape((128,))(y)

    # Combine audio and face features
    combined_input = concatenate([audio_output, face_output])
    z = Dense(64, activation="relu")(combined_input)
    z = Dropout(0.5)(z)

    # Separate outputs for laughter and smile detection
    laughter_output = Dense(2, activation="softmax", name='laughter_output')(z)
    smile_output = Dense(2, activation="softmax", name='smile_output')(z)

    # Create model
    model = Model(inputs=[audio_input, face_input], outputs=[laughter_output, smile_output])

    return model

input_shape_audio = (X_train_audio.shape[1], X_train_audio.shape[2])
input_shape_face = (X_train_face.shape[1], X_train_face.shape[2])

model = create_multimodal_model(input_shape_audio, input_shape_face)

# Specify evaluation metrics for each output
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics={
                  'laughter_output': ['accuracy'],
                  'smile_output': ['accuracy']
              })

#Train model
model.fit([X_train_audio, X_train_face], [y_train_audio, y_train_face], epochs=50, batch_size=32)

# Evaluate the model on the test set
score = model.evaluate([X_test_audio, X_test_face], [y_test_audio, y_test_face])
print("Evaluation Scores:", score)
print("Number of Evaluation Scores:", len(score))

#Print the accuracy and other metrics of model output
if len(score) == 4:  # Make sure there are 4 elements in the score list
    print("Test Loss:", score[0])
    print("Test Accuracy for Laughter Detection:", score[1])
    print("Test Accuracy for Smile Detection:", score[3])
else:
    print("The score list does not contain the expected number of elements. Please check the metric configuration.")


save_model(model, 'laughter_smile_detection_model.keras')

#Print model architecture
model.summary()

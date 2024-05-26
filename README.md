# SmileRecognition
SmileRecognition is a Python-based software designed to extract smile features (video frames, smile locations, smile rates) and laughter features (frequency range, energy distribution, time domain features) from videos. Utilizing convolutional neural networks, this tool detects and recognizes smiling faces and laughter, aiding research in psychology, human-computer interaction (HCI), and emotion analysis.

## Table of Contents
- [Features](#features)
- [Functions](#functions)
- [Installation](#installation)
- [Usage](#usage)
  - [Detecting Smiles in Videos](#detecting-smiles-in-videos)
  - [Analyzing Laughter in Audio](#analyzing-laughter-in-audio)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- **Smile Detection**: Detects and analyzes smiles in video frames.
- **Laughter Analysis**: Extracts laughter features from audio files.
- **High Accuracy**: Utilizes advanced convolutional neural networks.
- **Comprehensive Analysis**: Provides smile rates and detailed laughter features.
- **Open Source**: Fully open-source project under the MIT license.

## Functions
- **Development Principles and Design Philosophy
The software divides smiles into two dimensions: laughter and smiling faces, and extracts features using deep learning technology to establish a deep fusion convolutional neural network.
- **Specific Modules
  - **1. app Module
    main.py: Main program entry.
  - **2. data Module
    image, page, test, train: Resource files.
  - **3. interface_design Module
    ui_design.py: UI design of the software.
  - **4. interface_operation Module
    recognize_video.py: Implementation of recognition button.
    analyze_action.py: Implementation of analysis button.
    FileDownload.py: Implementation of file downloading.
    send_email.py: Implementation of sending emails.
  - **5. laughter_recognition Module
    **video_processing.py: Video data processing and feature extraction.
    **audio_processing.py: Audio data processing and feature extraction.
    **Deepfusion_network_model.py: Model establishment, training, and evaluation.
    **Smile Recognition：Utilizes pre-existing deep learning discriminators in OpenCV to identify facial grayscale values.Features include facial features and Smile Ratio (angle of mouth corner uplift).

- **Laughter Recognition
  - **Main features include Main Frequencies, Energy Distribution, and Time Domain Features, extracted using the librosa library.
  - **Deep Fusion Network Model
    Consists of three steps:
      Feature extraction and preprocessing.
      Model architecture: Utilizes Conv1D and LSTM layers to extract audio and facial features, followed by GlobalMaxPooling1D for global features.           Features are concatenated and passed through fully connected layers for classification.
      Training and evaluation: Compiles the model using the Adam optimizer, adds Recall and Precision evaluation metrics. Trains the model on the training set and evaluates it on the test set, outputting classification reports and confusion matrices.

## Installation
To install SmileRecognition, clone this repository and install the required dependencies:

```sh
git clone https://github.com/yourusername/SmileRecognition.git
cd SmileRecognition
pip install -r requirements.txt

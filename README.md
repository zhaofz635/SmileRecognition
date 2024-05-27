# SmileRecognition
SmileRecognition is a Python-based software designed to extract smile features (video frames, smile locations, smile rates) and laughter features (frequency range, energy distribution, time domain features) from videos. Utilizing convolutional neural networks, this tool detects and recognizes smiling faces and laughter, aiding research in psychology, human-computer interaction (HCI), and emotion analysis.

## Table of Contents
- [Features](#features)
- [Functions](#functions)
- [Installation](#installation)
- [Usage](#usage)
  - [Software idenification process](#Software identification process)
  - [Software optimization tips](#Software optimization tips)
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
- Development Principles and Design Philosophy
The software divides smiles into two dimensions: laughter and smiling faces, and extracts features using deep learning technology to establish a deep fusion convolutional neural network.
- Specific Modules
  - 1. app Module
    main.py: Main program entry.
  - 2. data Module
    image, page, test, train: Resource files.
  - 3. interface_design Module
    ui_design.py: UI design of the software.
  - 4. interface_operation Module
    - recognize_video.py: Implementation of recognition button.
    - analyze_action.py: Implementation of analysis button.
    - FileDownload.py: Implementation of file downloading.
    - send_email.py: Implementation of sending emails.
  - 5. laughter_recognition Module
    - video_processing.py: Video data processing and feature extraction.
    - audio_processing.py: Audio data processing and feature extraction.
    - Deepfusion_network_model.py: Model establishment, training, and evaluation.
    - Smile Recognition：Utilizes pre-existing deep learning discriminators in OpenCV to identify facial grayscale values.Features include facial features and Smile Ratio (angle of mouth corner uplift).

- Laughter Recognition
  - Main features include Main Frequencies, Energy Distribution, and Time Domain Features, extracted using the librosa library.
  - Deep Fusion Network Model
  - DConsists of three steps:
      - Feature extraction and preprocessing.
      - Model architecture: Utilizes Conv1D and LSTM layers to extract audio and facial features, followed by GlobalMaxPooling1D for global features.           Features are concatenated and passed through fully connected layers for classification.
      - Training and evaluation: Compiles the model using the Adam optimizer, adds Recall and Precision evaluation metrics. Trains the model on the training set and evaluates it on the test set, outputting classification reports and confusion matrices.


## Installation

To get started with SmileRecognition, follow these steps to install the necessary dependencies and set up the project.

### Prerequisites

- Python 3.7 or higher
- Git

### Required Libraries

The required libraries are listed in the `requirements.txt` file. These libraries include:

- numpy
- opencv-python
- dlib
- tensorflow
- scikit-learn

### Installing the Software

1. **Clone the Repository**

   First, clone the repository from GitHub:

   ```sh
   git clone https://github.com/yourusername/SmileRecognition.git
   cd SmileRecognition
   
2. **Create a Virtual Environment (Optional but Recommended)**
   
    it's a good practice to create a virtual environment to manage dependencies:
   ```sh
   Copy code
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install Dependencies**
   
   Install the required libraries using pip:
   ```sh
   Copy code
   pip install -r requirements.txt

4. **Downloading Data**

   If your software requires specific datasets, provide instructions on how to download them. For example:
   Download the Sample Video Data
   You can download the sample video data from this link. After downloading, unzip the files into a directory named data.
   ```sh
   Copy code
   mkdir data
   unzip sample-videos.zip -d data

## Usage

  Here is a guide on how to use the software:

1. **Software idenification process**
  - Navigate to `smile/app/main.py` and launch the software interface.
  - In the center of the interface, there are two core operation buttons: `Identify` and `Analysis`.
  - Recognition Process:
     - Click on `Identify` and select the audio or video file you want to recognize, formats such as `mp4` or `avi`.
     - Wait for the recognition process to complete. After completion, the `Identify` window will automatically close.
     - Click on the `Analysis` button to view the analysis results in `csv` format.
  - Downloading Analysis Results:
     - In the software's menu bar (including `Introduction`, `Contact Us`, `File Download`), click on `File Download` and choose the save path and input file name.

1. **Software optimization tips**
     - You can optimize the training data for the model and adjust the logic and code for feature extraction to further update and optimize the recognition software.
     

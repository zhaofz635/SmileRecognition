import os
import cv2
import pandas as pd
import numpy as np

# Load deep learning model for face detection
prototxt_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/deploy.prototxt"
model_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/res10_300x300_ssd_iter_140000.caffemodel"

# Make sure to provide the correct path
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Smiley face processing stage

# Define the path of the smile detector (XML file path of Haar cascade classifier)
smile_cascade_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/haarcascade_smile.xml"

# Load smile detector
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# Confirm whether the model file is loaded successfully
if smile_cascade.empty():
    raise IOError("Failed to load haarcascade_smile.xml. Please check the file path.")

# Process MP4 video files and save the results to DataFrame
def process_mp4_video(video_path, smile_threshold=0.1):
    # Read video number
    video_number = os.path.basename(video_path)[:-4]  # Assume that the video file name format is "video_number.mp4"

   #Load video file
    cap = cv2.VideoCapture(video_path)
    # Check if the video is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video:", video_path)
        return pd.DataFrame()

    # 初始化 DataFrame 列表
    rows = []

    #Loop to read video frames
    frame_count = 0
    while True:
        # Read a frame
        ret, frame = cap.read()
        # Check whether the frame was read successfully
        if not ret:
            break

        frame_count += 1

        # Convert frames to grayscale images (smiley face detector requires grayscale images as input)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detection
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        face_detected = False
        smile_detected = False
        face_position = None

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                face_detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face_position = (startX, startY, endX - startX, endY - startY)
                face = gray_frame[startY:endY, startX:endX]

                # Detect smiling faces on grayscale images
                smiles = smile_cascade.detectMultiScale(
                    face,
                    scaleFactor=1.8,  # Increase scaleFactor to improve detection sensitivity
                    minNeighbors=20,
                    minSize=(15, 15),  # Reduce minSize to detect smaller smiley faces
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (sx, sy, sw, sh) in smiles:
                    # Calculate the ratio of mouth height to face height
                    mouth_height = sh
                    face_height = endY - startY

                    smile_ratio = mouth_height / face_height

                    # Output debugging information
                    # print(f"Frame {frame_count}: smile_ratio = {smile_ratio}, threshold = {smile_threshold}")

                    # If the ratio of the height of the mouth to the height of the face is greater than the threshold, it is considered that a smile is detected
                    if smile_ratio > smile_threshold:
                        rows.append([video_number, frame_count, (startX + sx, startY + sy, sw, sh), smile_ratio, True])
                        # print(f"Smile detected in frame {frame_count} at position {(startX + sx, startY + sy, sw, sh)}")
                        smile_detected = True
                        # Draw a rectangular box
                        # cv2.rectangle(frame, (startX + sx, startY + sy), (startX + sx + sw, startY + sy + sh), (0, 255, 0), 2)
                        # # Draw text
                        # cv2.putText(frame, 'Smile', (startX + sx, startY + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        break

                if not smile_detected:
                    rows.append([video_number, frame_count, face_position, None, False])
                    # print(f"No smile detected in frame {frame_count} in face region {face_position}")
                break

        if not face_detected:
            rows.append([video_number, frame_count, None, None, False])
            # print(f"No face detected in frame {frame_count}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream
    cap.release()
    cv2.destroyAllWindows()

    #Create DataFrame
    df = pd.DataFrame(rows, columns=['Video Number', 'Frame Number', 'Face Position', 'Smile Ratio', 'Detected'])
    return df


# Define MP4 video folder path list
mp4_videos_folders = [
    "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train",
    "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test"]

# Iterate through each folder and its subfolders, process each video and save the results to a DataFrame
dfs = []
for folder in mp4_videos_folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                mp4_video_path = os.path.join(root, file)
                df = process_mp4_video(mp4_video_path, smile_threshold=0.1)  
                if not df.empty:
                    dfs.append(df)


# Merge all DataFrames
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)

    # Use logic to populate the 'Smile Ratio' column
    final_df['Smile Ratio'] = final_df.apply(lambda x: x['Smile Ratio'] if x['Detected'] else 0.1 if x['Smile Ratio'] is not None else 0, axis=1)

    # Check the final DataFrame columns
    print(final_df.columns)

    final_df.to_csv("output.csv", index=False)
else:
    print("No MP4 videos found in the specified folders.")

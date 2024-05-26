import os
import cv2
import pandas as pd
import numpy as np

# 加载深度学习模型进行面部检测
prototxt_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/deploy.prototxt"
model_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/res10_300x300_ssd_iter_140000.caffemodel"

# 确保提供正确的路径
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# 笑脸处理阶段

# 定义笑脸检测器的路径（Haar 级联分类器的 XML 文件路径）
smile_cascade_path = "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/laughter_recognition/haarcascade_smile.xml"

# 加载笑脸检测器
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

# 确认模型文件是否加载成功
if smile_cascade.empty():
    raise IOError("Failed to load haarcascade_smile.xml. Please check the file path.")

# 处理 MP4 视频文件并保存结果到 DataFrame
def process_mp4_video(video_path, smile_threshold=0.1):
    # 读取视频编号
    video_number = os.path.basename(video_path)[:-4]  # 假设视频文件名格式为"video_number.mp4"

    # 加载视频文件
    cap = cv2.VideoCapture(video_path)
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open video:", video_path)
        return pd.DataFrame()

    # 初始化 DataFrame 列表
    rows = []

    # 循环读取视频帧
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 检查是否成功读取帧
        if not ret:
            break

        frame_count += 1

        # 将帧转换为灰度图像（笑脸检测器需要灰度图像作为输入）
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 面部检测
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

                # 在灰度图像上检测笑脸
                smiles = smile_cascade.detectMultiScale(
                    face,
                    scaleFactor=1.8,  # 增加 scaleFactor 以提高检测的灵敏度
                    minNeighbors=20,
                    minSize=(15, 15),  # 减小 minSize 以检测较小的笑脸
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (sx, sy, sw, sh) in smiles:
                    # 计算嘴巴高度占脸部高度的比例
                    mouth_height = sh
                    face_height = endY - startY

                    smile_ratio = mouth_height / face_height

                    # 输出调试信息
                    # print(f"Frame {frame_count}: smile_ratio = {smile_ratio}, threshold = {smile_threshold}")

                    # 如果嘴巴高度占脸部高度的比例大于阈值，则认为检测到笑脸
                    if smile_ratio > smile_threshold:
                        rows.append([video_number, frame_count, (startX + sx, startY + sy, sw, sh), smile_ratio, True])
                        # print(f"Smile detected in frame {frame_count} at position {(startX + sx, startY + sy, sw, sh)}")
                        smile_detected = True
                        # 绘制矩形框
                        # cv2.rectangle(frame, (startX + sx, startY + sy), (startX + sx + sw, startY + sy + sh), (0, 255, 0), 2)
                        # # 绘制文字
                        # cv2.putText(frame, 'Smile', (startX + sx, startY + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        break

                if not smile_detected:
                    rows.append([video_number, frame_count, face_position, None, False])
                    # print(f"No smile detected in frame {frame_count} in face region {face_position}")
                break

        if not face_detected:
            rows.append([video_number, frame_count, None, None, False])
            # print(f"No face detected in frame {frame_count}")

        # 可视化中间结果
        # for (sx, sy, sw, sh) in smiles:
        #     # 绘制矩形框
        #     cv2.rectangle(frame, (startX + sx, startY + sy), (startX + sx + sw, startY + sy + sh), (0, 255, 0), 2)
        #     # 绘制文字
        #     cv2.putText(frame, 'Smile', (startX + sx, startY + sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频流
    cap.release()
    cv2.destroyAllWindows()

    # 创建 DataFrame
    df = pd.DataFrame(rows, columns=['Video Number', 'Frame Number', 'Face Position', 'Smile Ratio', 'Detected'])
    return df


# 定义 MP4 视频文件夹路径列表
mp4_videos_folders = [
    "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/train",
    "/Users/fuzhengzhao/PycharmProjects/pythonProject/simle/data/test"]

# 遍历每个文件夹及其子文件夹，处理每个视频并保存结果到 DataFrame
dfs = []
for folder in mp4_videos_folders:
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.mp4'):
                mp4_video_path = os.path.join(root, file)
                df = process_mp4_video(mp4_video_path, smile_threshold=0.1)  # 降低阈值以提高检测率
                if not df.empty:
                    dfs.append(df)


# 合并所有 DataFrame
if dfs:
    final_df = pd.concat(dfs, ignore_index=True)

    # 使用逻辑填充'Smile Ratio'列
    final_df['Smile Ratio'] = final_df.apply(lambda x: x['Smile Ratio'] if x['Detected'] else 0.1 if x['Smile Ratio'] is not None else 0, axis=1)

    # 检查最终的 DataFrame 列
    print(final_df.columns)

    # 保存结果到 CSV 文件
    final_df.to_csv("output.csv", index=False)
else:
    print("No MP4 videos found in the specified folders.")

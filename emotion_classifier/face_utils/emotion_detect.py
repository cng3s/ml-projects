import sys
import cv2
import os
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.assistant import detect_faces, draw_text
from utils.assistant import draw_bounding_box, apply_offsets
from utils.assistant import load_detection_model, load_image
from utils.preprocessor import preprocess_input
from config.path_config import *


def emotion_detect(image_list):
    for img_path in image_list:
        # 图片路径
        save_path = config_output_basic_path + img_path.split('/')[-1]
        # image_path = '../datasets/mini_faces/file0001.jpg'
        if img_path == None or os.path.exists(img_path) is False:
            print("image_path路径错误: {}".format(img_path))
            continue

        # 设置人脸识别模型，情绪识别模型，性别识别模型的路径
        detection_model_path = config_detection_model_path
        emotion_model_path = config_emotion_model_path
        gender_model_path = config_gender_model_path

        # 加载fer2013标签作为情绪识别标签, imdb标签作为性别识别标签
        emotion_labels = get_labels('fer2013')
        gender_labels = get_labels('imdb')
        # 加载模型
        face_detection = load_detection_model(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)
        gender_classifier = load_model(gender_model_path, compile=False)
        # 输入模型大小
        emotion_target_size = emotion_classifier.input_shape[1:3]
        gender_target_size = gender_classifier.input_shape[1:3]
        # 加载图片
        rgb_image = load_image(img_path, grayscale=False)
        gray_image = load_image(img_path, grayscale=True)
        # 删除一维项
        gray_image = np.squeeze(gray_image)
        # 类型转换为uint8
        gray_image = gray_image.astype('uint8')
        # 人脸框盒子的位置设置
        gender_offsets = (10, 10)
        emotion_offsets = (0, 0)


        # 检测人脸
        faces = detect_faces(face_detection, gray_image)
        for face_coordinates in faces:
            x1, x2, y1, y2 = apply_offsets(face_coordinates, gender_offsets)
            rgb_face = rgb_image[y1:y2, x1:x2]
            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]

            try:
                rgb_face = cv2.resize(rgb_face, gender_target_size)
                gray_face = cv2.resize(gray_face, emotion_target_size)
            except:
                continue
            
            rgb_face = preprocess_input(rgb_face, False) # 将矩阵值变化到[0, 1]，避免大量计算产生溢出
            rgb_face = np.expand_dims(rgb_face, 0)
            gender_prediction = gender_classifier.predict(rgb_face)
            gender_label_arg = np.argmax(gender_prediction)
            gender_text = gender_labels[gender_label_arg]

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]

            # if gender_text == gender_labels[0]:
            #     color = (0, 0, 255)
            # else:
            #     color = (255, 0, 0)

            if emotion_text == 'angry':
                color = emotion_probability * np.asarray((255, 0, 0))
            elif emotion_text == 'sad':
                color = emotion_probability * np.asarray((0, 0, 255))
            elif emotion_text == 'happy':
                color = emotion_probability * np.asarray((255, 255, 0))
            elif emotion_text == 'surprise':
                color = emotion_probability * np.asarray((0, 255, 255))
            else:
                color = emotion_probability * np.asarray((0, 255, 0))

            draw_bounding_box(face_coordinates, rgb_face, color)
            draw_text(face_coordinates, rgb_image, gender_text, color, 50, 100, 1, 2)
            draw_text(face_coordinates, rgb_image, emotion_text, color, 50, 140, 1, 2)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        print(save_path)
        cv2.imwrite(save_path, bgr_image)


if __name__ == '__main__':
    #emotion_detect()
    pass
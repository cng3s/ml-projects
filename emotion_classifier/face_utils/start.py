""" 人脸情绪分类器 """
import argparse
import os
from Face import Face
from emotion_detect import emotion_detect
from train_emotion_classifier import train_emotion_classifier
from train_gender_classifier import train_gender_classifier
from video_emotion_detect import video_emotion_detect
from utils.preprocessor import generate_images_list

face = Face()
function_table = {
    'detect': face.detect,                                        # 人脸检测
    'cnn_detect': face.cnn_detect,                                # 基于CNN的人脸检测
    'landmark_detect': face.landmark_detect,                      # 人脸特征点检测
    'recognition': face.recognition,                              # 人脸识别
    'alignment': face.alignment,                                  # 人脸对齐
    'clustering': face.clustering,                                # 人脸聚类
    'jitter': face.jitter,                                        # 人脸抖动/增强
    'emotion_detect': emotion_detect,                             # 人脸情绪检测
    'emotion_classifier': train_emotion_classifier,               # 训练情绪分类器
    'gender_classifier': train_gender_classifier,                 # 训练性别分类器
    'video_emotion_detect': video_emotion_detect,                 # 视频实时情绪检测
}

def main():
    parser = argparse.ArgumentParser("人脸情绪识别分类器")
    parser.add_argument("--file", type=str, help="指定需要探测人脸图片文件名")
    parser.add_argument("--folder", type=str, help="指定需要探测人脸图片的文件夹")
    parser.add_argument("--method", type=str, default=None, 
        help="指定所使用的方法:[METHOD: detect, cnn_detect, emotion_detect, recognition, alignment, clustering, jitter]")
    parser.add_argument("--train", type=str, default=None,
        help="训练分类器:[TRAINTER: gender_classifier, emotion_classifier]")
    parser.add_argument("--video", type=str, default=None,
        help="视频检测模块:[VIDEO: video_emotion_detect]")
    args = parser.parse_args()

    image_list = []
    if args.folder is not None:
        image_list = generate_images_list(args.folder)
    elif args.file is not None:
        image_list.append(args.file)

    method = args.method
    if method != None:
        func = function_table[method]
        func(image_list)
    
    train = args.train
    if train != None:
        func = function_table[train]
        func()

    video = args.video
    if video != None:
        func = function_table[video]
        func()
    

if __name__ == '__main__':
    main()
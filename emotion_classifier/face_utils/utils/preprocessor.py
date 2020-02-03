import numpy as np
import os
from scipy.misc import imread, imresize


def generate_images_list(folder_path):
    """ 传入文件夹路径，遍历文件夹生成文件列表images_list """
    image_list = []
    for root, _, files in os.walk(folder_path, topdown=False):
        for file in files:
            image_list.append(os.path.join(root, file))
    return image_list


def preprocess_input(x, v2=True):
    ''' 对输入矩阵进行预处理到[0, 1]之间 避免溢出和大量运算 '''
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def img_read(image_name):
    return imread(image_name)

def to_categorical(integer_classes, num_classes=2):
    integer_classes = np.asarray(integer_classes, dtype='int')
    num_samples = integer_classes.shape[0]
    categorical = np.zeros((num_samples, num_classes))
    categorical[np.arange(num_samples), integer_classes] = 1
    return categorical
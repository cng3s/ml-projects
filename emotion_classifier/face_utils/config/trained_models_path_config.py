"""
训练好的模型路径配置文件，可以直接在这里修改模型路径
"""

basic_path = '../datasets/trained_models/'
config_shape_predictor_path = basic_path + 'shape_predictor_65_face_landmarks.dat'
config_cnn_detector_path = basic_path + 'mmod_human_face_detector.dat'
config_recognition_model_path = basic_path + 'dlib_face_recognition_resnet_model_v1.dat'

# image_emotion.py中使用的人脸检测，情绪检测，性别检测模型路径
config_detection_model_path = basic_path + 'detection_models/haarcascade_frontalface_default.xml'
config_emotion_model_path = basic_path + 'emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
config_gender_model_path = basic_path + 'gender_models/simple_CNN.81-0.96.hdf5'
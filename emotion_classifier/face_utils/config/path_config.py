"""
训练集路径配置文件
"""
configs_faces_folder_path = '../datasets/images/'
configs_mini_faces_folder_path = '../datasets/mini_faces/'
configs_clustering_output_folder = '../output/clustering'

"""
训练好的模型路径配置文件，可以直接在这里修改模型路径
"""

config_models_basic_path = '../datasets/trained_models/'
config_shape_predictor_path = config_models_basic_path + 'shape_predictor_65_face_landmarks.dat'
config_cnn_detector_path = config_models_basic_path + 'mmod_human_face_detector.dat'
config_recognition_model_path = config_models_basic_path + 'dlib_face_recognition_resnet_model_v1.dat'

"""
image_emotion.py中使用的人脸检测，情绪检测，性别检测模型路径
"""
config_detection_model_path = config_models_basic_path + 'detection_models/haarcascade_frontalface_default.xml'
config_emotion_model_path = config_models_basic_path + 'emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
config_gender_model_path = config_models_basic_path + 'gender_models/simple_CNN.81-0.96.hdf5'

"""
文件保存基本路径
"""
config_output_basic_path = '../output/emotion_detect_results/'
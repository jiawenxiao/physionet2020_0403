#!/usr/bin/env python

import numpy as np
import joblib
from get_12ECG_features import get_12ECG_features
from tensorflow.keras.models import load_model
import keras 
import tensorflow
# from keras.models import load_model
from keras.models import model_from_json

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def run_12ECG_classifier(data,header_data,classes,model):
    
    num_classes = len(classes)
    current_label = np.zeros(num_classes, dtype=int)
    current_score = np.zeros(num_classes)
    # Use your classifier here to obtain a label and score for each class. 
    feats_reshape,feats_external = get_12ECG_features(data,header_data)
    
    tmp_score = model.predict([feats_reshape,feats_external])  #输出维度(1,9)
    
    tmp_label = np.where(tmp_score>0.12,1,0)
    
    for i in range(num_classes):
        current_label[i] = np.array(tmp_label[0][i])
        current_score[i] = np.array(tmp_score[0][i])

    return current_label, current_score

def load_12ECG_model():
    # load the model from disk 
#     filename='physionet_cnn_0403.h5'
#     loaded_model = load_model(filename)

    with open("model_save_0403.json", "r") as f:
        json_string = f.read()  # 读取本地模型的json文件
    model = model_from_json(json_string)  # 创建一个模型
    model.load_weights("model_weight_0403.h5")

    return model

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.transform import resize
import os
import random
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import keras
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import tensorflow as tf

def extract_hypercolumn(model, layer_indexes, instance,image_size):
    # Belirtilen katmanların çıktısını almak için yeni model oluştur
    layers = [model.layers[li].output for li in layer_indexes]
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=layers)

    # Özellik haritalarını al
    feature_maps = feature_extractor(instance)

    hypercolumns = []

    for convmap in feature_maps:
        if len(convmap.shape) > 2:
            conv_out = convmap[0].numpy()  # TensorFlow tensor'ünü numpy array'e çevir
            feat_map = np.transpose(conv_out, (2, 0, 1))  # Kanal boyutunu öne al
            for fmap in feat_map:
                upscaled = resize(fmap, image_size, mode='constant', preserve_range=True)
                hypercolumns.append(upscaled)
        else:
            for i in range(convmap.shape[-1]):
                upscaled = np.full(image_size, convmap[0, i].numpy())  # TensorFlow tensor'ünü numpy'e çevir
                hypercolumns.append(upscaled)

    return np.asarray(hypercolumns)

def image_feature_extract(model,image_path,image_size):
  layers_extract = [2,5,9,13,17]

  img = image.load_img(image_path , target_size=(image_size, image_size,3))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  hc = extract_hypercolumn(model, layers_extract, x,(image_size,image_size))
  hc = hc.reshape(1472,(image_size*image_size)).transpose(1,0)
  return hc
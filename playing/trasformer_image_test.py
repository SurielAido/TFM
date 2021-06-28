import numpy as np
from keras_transformer import get_model, decode
import keras as k
from keras.preprocessing.image import ImageDataGenerator
import scipy

dataset = []
data_training = './data/entrenamiento'
data_validation = './data/validacion'

entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_training,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical')

print('stop')

# No voy a separar entre imágenes; aqui no tengo dos "idiomas".

def build_token_dict(dataset):
    token_dict = {
        'PAD': 0,
        '<START>': 1,
        'END': 2
    }
    for images in dataset:
        for frame in images:
            if frame not in token_dict:
                token_dict[frame] = len(frame) * imagen_entrenamiento
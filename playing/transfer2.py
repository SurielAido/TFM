from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.loading_dataset import LoadingDataset
import numpy as np
import time

'''
    Inicializamos el modelo. Con los pesos del dataset Imagenet. El include_top=False indica que no incluyamos
    las capas densas, solos las de convolución. En input_shape indica en qué formato vamos a introducir las 
    imágenes (150 alto, 150 ancho, RGB)
'''
model_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
''' Indicamos la carpeta de entrenamiento'''
base_dir = '../core/dataset/train'
# model_base.trainable = False
datagen = ImageDataGenerator(1. / 255)
'''Imágenes que va a pasar por cad a'iteración'''
batch_size = 100

model_base.summary()


def extract_features(directory, sample_count):
    '''
        np.zeros inicializa un array de numpy con 0s. Si se le mete una tupla en lugar de un
        número, incializa una matriz de esas dimensiones. En este caso se van a inicializar
        el de features, con tamaño (nº ejemplos, 3, 3 ,512, donde 3, 3 y 512 son las dimensiones
        de la última capa del modelo con el include_top = false (véase eñ summary()).
    '''
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count, 51))
    '''
        flow_from_directory indicará que ese datagen lo aplique en la carpeta especificada
        con una resolución de 112x112 y que clasifique en categorías, con el número de imágenes
        a entrenar en cada bloque (batch_size)
    '''
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical'
    )
    i = 0
    start = time.time()
    total_start = time.time()
    for inputs_batch, labels_batch in generator:
        # Genera la clase predecida con el ejemplo introducido en el input
        timecheckpoint = time.time()
        features_batch = model_base.predict(inputs_batch)

        features[i * batch_size: (i + 1) * batch_size] = features_batch  # i[0: 100] = predicción
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch  # i[i*100: (i+1)*100 = clase
        i += 1
        if (timecheckpoint - start) >= 300:
            print("Sigo aqui, funcionando.")
            start = time.time()
        if i * batch_size >= sample_count: # Si i * batch es mayor que el número de ejemplos que le he pasado, para.
            break
    total_end = time.time()
    print(total_end - total_start)
    return features, labels


train_features, train_labels = extract_features(base_dir, 34657)
train_features = np.reshape(train_features, (34657, 4 * 4 * 512))
model = models.Sequential()
model.add(Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(Dropout(0.5))
model.add(Dense(51, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_features, train_labels, epochs=500, batch_size=batch_size, validation_split=0.2)

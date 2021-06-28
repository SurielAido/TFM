import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, save_model
from keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D
from keras import backend

data_training = '../core/dataset/train'    # Dataset entrenamiento
data_validation = '../core/validation/images_test'  # Dataset validación


#Parámetros.
epochs = 500  # Número de veces que vamos a iterar sobre el dataset en nuestro entrenamiento
altura, longitud = 100, 100  # Tamaño de procesamiento de imágenes (limitamos el tamaño a 100x100px)
batch_size = 32  # Número de imágenes que se va a procesar en cada paso.
pasos = 1000  # Número de veces que se va a procesar la información por cada época.
pasos_validacion = 200  # Al final de cada época se va a correr 200 veces sobre el dataset de datos de validación
filtrosConv1 = 32  # Número de filtros que se van a aplicar en cada convolución
filtrosConv2 = 64  # Número de filtros que se van a aplicar en cada convolución
# La imagen va a tener una profundidad de 32 tras la primera conv y de 64 tras la 2ª
tamano_filtro1 = (3, 3)  # Tamaño matriz (3x3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)  # Tamaño max_pooling
clases = 51  # Gato, perro, gorila
lr = 0.0005  # Learning rate.

# Images preprocessing

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Cada uno de nuestros píxeles tiene un rango entre 0 y 1 en lugar de entre 1 y 255
    shear_range=0.3,
    # Genera el 0,3 de nuestras imágenes ligeramente inclinada, para que el algoritmo entienda que un perro no siempre está perfectamente vertical
    zoom_range=0.3,  # Le va a hacer zoom al 0,3 de nuestras imágenes.
    horizontal_flip=True)  # va a invertir algunas imágnees.

validation_datagen = ImageDataGenerator(rescale=1. / 255)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_training,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

# Esto lo que va a hacer es entrar a la carpeta de entrenamiento, las va a ajustar
# a la altura y longitud específica que le hemos pasado, le va a pasar el batch
# definido antes y lo vamos a etiquetar de manera categórica (perro, gato, gorila).

imagen_validacion = validation_datagen.flow_from_directory(
    data_validation,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')

## CREAMOS NUESTRA CNN ###

cnn = Sequential()  # Nuestra red es secuencial (varias capas apiladas entre ellas).
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same',
                      input_shape=(altura, longitud, 3),
                      activation='relu'))  # input_shape van a tener una altura y ogitud específicas
cnn.add(MaxPooling2D(
    pool_size=tamano_pool))  # El max pooling lo que hace es decirle que la siguiente capa va a ser la de maño tamano.
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())  # Vamos a hacer la imagen (profunda y pequeña) una imagen plana.
cnn.add(
    Dense(256, activation='relu'))  # aquí lo que hacemos es coger la imagen "plana" de antes a una capa full-connected.
cnn.add(Dropout(
    0.5))  # Aquí lo que hacemos es apagarle el 50% de las neuronas a nuestra capa densa por paso (para evitar sobreajustar).
cnn.add(Dense(clases,
              activation='softmax'))  # Softmax lo que hace es 80% de que sea un perro, 10% que sea un gato, 10% que sea un gorila (p.e).

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])
# Con esto acemos que durante entrenamiento la función de pérdida es 'categorical_...', con ese optimizados y esa métrica.

cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epochs,
        validation_data=imagen_validacion, validation_steps=pasos_validacion)

dir = '../playing/modelo'
if not os.path.exists(dir):
     os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')
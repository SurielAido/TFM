from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Convolution2D, MaxPooling2D


class MyConvolutionalNetwork:
    def __init__(self, epochs, width, height, batch_size, steps, validation_steps, conv_filters1, conv_filters2,
                 filter_size1, filter_size2, pool_size, number_classes, learning_rate,
                 data_training='../core/dataset/train', data_validation='../core/validation/images_test'):
        self.epochs = epochs
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.steps = steps
        self.validation_steps = validation_steps
        self.conv_filters1 = conv_filters1
        self.conv_filters2 = conv_filters2
        self.filter_size1 = filter_size1
        self.filter_size2 = filter_size2
        self.pool_size = pool_size
        self.number_classes = number_classes
        self.learning_rate = learning_rate
        self.data_training = data_training
        self.data_validation = data_validation

    @staticmethod
    def datagenerator(self):
        return ImageDataGenerator(
            validation_split=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

    def training_generator(self):
        return self.datagenerator().flow_from_directory(
            self.data_training,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='categorical'
        )

    def validation_generator(self):
        return self.datagenerator().flow_from_directory(
            self.data_training,
            target_size=(self.height, self.width),
            batch_size=self.batch_size,
            class_mode='categorical')

    def create_model(self):
        cnn = Sequential()  # Nuestra red es secuencial (varias capas apiladas entre ellas).
        cnn.add(Convolution2D(self.conv_filters1, self.filter_size1, padding='same',
                              input_shape=(self.height, self.width, 3),
                              activation='relu'))  # input_shape van a tener una altura y ogitud específicas
        cnn.add(MaxPooling2D(
            pool_size=self.pool_size))  # El max pooling lo que hace es decirle que la siguiente capa va a ser la de maño tamano.
        cnn.add(Convolution2D(self.conv_filters2, self.filter_size2, padding='same', activation='relu'))
        cnn.add(MaxPooling2D(pool_size=self.pool_size))

        cnn.add(Flatten())  # Vamos a hacer la imagen (profunda y pequeña) una imagen plana.
        cnn.add(
            Dense(256,
                  activation='relu'))  # aquí lo que hacemos es coger la imagen "plana" de antes a una capa full-connected.
        cnn.add(Dropout(
            0.5))  # Aquí lo que hacemos es apagarle el 50% de las neuronas a nuestra capa densa por paso (para evitar sobreajustar).
        cnn.add(Dense(self.number_classes,
                      activation='softmax'))  # Softmax lo que hace es 80% de que sea un perro, 10% que sea un gato, 10% que sea un gorila (p.e).

        cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=self.learning_rate),
                    metrics=['accuracy'])
        # Con esto acemos que durante entrenamiento la función de pérdida es 'categorical_...', con ese optimizados y esa métrica.
        return cnn

    def train_model(self):
        cnn = self.create_model()
        cnn.fit(self.training_generator(), steps_per_epoch=self.steps, epochs=self.epochs,
                validation_data=self.validation_generator(), validation_steps=self.validation_steps)
        destination_model_cnn = 'model/cnn/cnn_trained.h5'
        save_model(cnn, destination_model_cnn)


# CNN = MyConvolutionalNetwork(100, 112, 112, 10, 500, 100, 32, 64, (3, 3), (2, 2), (2, 2), 51, 0.0005)
# CNN.train_model()

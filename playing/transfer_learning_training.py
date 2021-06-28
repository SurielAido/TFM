import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from core.TrainVITModel import Patches, PatchEncoder, patch_size, num_patches, projection_dim, transformer_layers, \
    num_heads, mlp, transformer_units, mlp_head_units, data_augmentation, x_train, y_train, x_test, y_test

train_data_dir = '../core/dataset/train'
validation_data_dir = '../core/dataset/validation/images_test'


class TransferLearningTraining:

    def __init__(self, width_image, height_image, dimensions, num_classes, epochs, batch_size,
                 train_samples, validation_samples):
        self.width_image = width_image
        self.height_image = height_image
        self.dimensions = dimensions
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_samples = train_samples
        self.validation_samples = validation_samples

    def train_datagen(self):
        return ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            preprocessing_function=imagenet_utils.preprocess_input)

    def valid_datagen(self):
        return ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
            preprocessing_function=imagenet_utils.preprocess_input)

    def train_generator(self):
        return self.train_datagen().flow_from_directory(
            train_data_dir,
            target_size=(self.width_image, self.height_image),
            batch_size=self.batch_size,
            class_mode='categorical')

    def validation_generator(self):
        validation_generator = self.valid_datagen().flow_from_directory(
            validation_data_dir,
            target_size=(self.width_image, self.height_image),
            batch_size=self.batch_size,
            class_mode='categorical')
        return validation_generator

    def train_VGG16(self):
        image_input = layers.Input(batch_size=self.batch_size,
                                   shape=(self.width_image, self.height_image, self.dimensions))
        model = VGG16(input_tensor=image_input, include_top=False, weights='imagenet')
        for layer in model.layers:
            layer.trainable = False
        # model.summary()
        # last_layer = model.get_layer('block5_pool').output
        # x = layers.Flatten(name='flatten2')(last_layer)
        # x = layers.Dense(128, activation='relu', name="fc1")(x)
        # x = layers.Dense(128, activation='relu', name="fc2")(x)
        # out = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        # model = Model(image_input, out)
        ####################################################################################################
        inputs = layers.Input(batch_size=self.batch_size, shape=(self.width_image, self.height_image, self.dimensions))
        # Aumentamos los datos
        augmented = data_augmentation(inputs)
        # Creamos los parches
        patches = Patches(patch_size)(augmented)
        # Codificamos los parches
        enconded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Creamos múltiples capas para el bloque del transformador
        for _ in range(transformer_layers):
            # Capa 1 de normalización
            x1 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
            # Creamos una capa de atención con múltiples cabeeras
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Saltamos la primera conexión
            x2 = layers.Add()([attention_output, enconded_patches])
            # Segunda capa de normalización
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Saltamos la segunda conexión
            enconded_patches = layers.Add()([x3, x2])

        # Creamos un tensor con el batch_size y la projection_dim pertinentes
        representation = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Añadimos la MLP
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Clasificamos la salida
        logits = layers.Dense(self.num_classes)(features)
        # Creamos el modelo de Keras.
        vit_model = Model(inputs=inputs, outputs=logits)
        ####################################################################################################

        # my_model = Model(image_input, out)
        # for layer in my_model.layers[:-3]:
        #     layer.trainable = False

        out1 = model.output
        out2 = vit_model.output
        out2 = out2[..., np.newaxis, np.newaxis]
        out2 = layers.Reshape((7, 7, 512))(out2)
        # out2 = layers.ZeroPadding2D(padding=2)(out2)
        print(out1)
        print('##################################################################################')
        print(out2)
        concatenated = layers.concatenate([out1, out2])

        # concatenated = layers.concatenate([model.input, vit_model.input])
        concatenated_out = layers.Dense(1, activation='softmax', name='output_layer')(concatenated)

        my_model = Model(inputs=[model.input, vit_model.input], outputs=[concatenated_out])
        # seq_model = Sequential()
        # seq_model.add(model)
        # seq_model.add(vit_model)
        # seq_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        my_model.fit([x_train, y_train], [y_train, x_train], batch_size=self.batch_size, epochs=self.epochs,
                     validation_data=([x_test, y_test], [y_test, x_test]))

        my_model.save('transfer_learning_models/vgg16/vgg16_model.h5')

    def train_ResNet50(self):
        image_input = layers.Input(shape=(self.width_image, self.height_image, self.dimensions))
        model = ResNet50(input_tensor=image_input, include_top=False, weights='imagenet')
        # model.summary()
        # vit_model = self.create_vit_classifier()
        # last_layer = model.layers[-1].output

        inputs = layers.Input(shape=(self.width_image, self.height_image, self.dimensions))
        augmented = data_augmentation(inputs)
        patches = Patches(patch_size)(augmented)
        enconded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        x1 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, enconded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        enconded_patches = layers.Add()([x3, x2])
        out = layers.Dense(self.num_classes, activation='softmax', name='output')(enconded_patches)
        #
        # x = layers.Flatten(name='flatten')(last_layer)
        # x = layers.Dense(128, activation='relu', name='fc1')(x)
        # x = layers.Dropout(0.3)(x)
        # x = layers.Dense(128, activation='relu', name='fc2')(x)
        # x = layers.Dropout(0.3)(x)
        # out = layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        model = Model(image_input, out)
        for layer in model.layers[:-7]:
            layer.trainable = False
        my_model = Model(image_input, out)
        my_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        model_ResNet50_trained = my_model.fit_generator(
            self.train_generator(),
            epochs=self.epochs,
            validation_data=self.validation_generator(),
            steps_per_epoch=self.train_samples // self.batch_size,
            validation_steps=self.validation_samples // self.batch_size
        )

        model_ResNet50_trained.save('transfer_learning_models/resnet50/resnet50_model.h5')

    def create_vit_classifier(self):
        print("Vamos a crear el clasificador...")
        inputs = layers.Input(shape=(self.width_image, self.height_image, self.dimensions))
        # Aumentamos los datos
        augmented = data_augmentation(inputs)
        # Creamos los parches
        patches = Patches(patch_size)(augmented)
        # Codificamos los parches
        enconded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Creamos múltiples capas para el bloque del transformador
        for _ in range(transformer_layers):
            # Capa 1 de normalización
            x1 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
            # Creamos una capa de atención con múltiples cabeeras
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Saltamos la primera conexión
            x2 = layers.Add()([attention_output, enconded_patches])
            # Segunda capa de normalización
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Saltamos la segunda conexión
            enconded_patches = layers.Add()([x3, x2])

        # Creamos un tensor con el batch_size y la projection_dim pertinentes
        representation = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Añadimos la MLP
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Clasificamos la salida
        logits = layers.Dense(self.num_classes)(features)
        # Creamos el modelo de Keras.
        model = Model(inputs=inputs, outputs=logits)
        return model


# width_image, height_image, dimensions, num_classes, epochs, batch_size, train_samples, validation_samples
tlf = TransferLearningTraining(224, 224, 3, 51, 100, 32, 1490, 50)
tlf.train_VGG16()
# tlf.train_ResNet50()
# tlf.create_vit_classifier()

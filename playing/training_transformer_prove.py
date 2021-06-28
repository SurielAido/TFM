import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from utils.loading_dataset import LoadingDataset

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Training:

    # def __init__(self, loader,  x_train, x_test, y_train, y_test, num_classes, learning_rate=0.001, weight_decay=0.0001,
    #              batch_size=256, num_epochs=100, image_size=72, patch_size=6, num_patches=144, projection_dim=64,
    #              ndum_heas=4, transformer_units=[128, 34], transformer_layers=8, mlp_head_units=[2048, 1024],
    #              input_shape=(32, 32, 3)):
    #     self.loader = LoadingDataset(input_shape[0], input_shape[1])
    #     self.input_shape = input_shape
    #     self.x_train = x_train
    #     self.x_test = x_test
    #     self.y_train = y_train
    #     self.y_test = y_test
    #     self.num_classes = num_classes
    #     self.x_train, self.x_test, self.y_train, self.y_test, self.num_classes = LoadingDataset.take_me_dataset(loader)
    #     self.learning_rate = learning_rate
    #     self.weight_decay = weight_decay
    #     self.batch_size = batch_size
    #     self.num_epochs = num_epochs
    #     self.image_size = image_size
    #     self.patch_size = patch_size
    #     self.num_patches = num_patches
    #     self.projection_dim = projection_dim
    #     self.ndum_heas = ndum_heas
    #     self.transformer_units = transformer_units
    #     self.transformer_layers = transformer_layers
    #     self.mlp_head_units = mlp_head_units

    input_shape = (32, 32, 3)
    loader = LoadingDataset(input_shape[0], input_shape[1])
    x_train, x_test, y_train, y_test, num_classes = LoadingDataset.take_me_dataset(loader)
    print('Anchura: ', str(input_shape[0]))
    print('Altura: ', str(input_shape[1]))
    print('Número de clases: ', str(num_classes))

    ############### Configuramos los hiperparámetros ###############
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = 72  # Vamos a redimensionar las imágenes de entrada a este tamaño
    patch_size = 6  # Tamaño de los "parches" que vamos a extraer de las imágenes de entrada
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    ndum_heas = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ]  # Tamaño de las capas del transformer
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Tamaño de las capas densas del clasificador final

    ############### Usamos el aumento de datos ###############
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    data_augmentation.layers[0].adapt(
        x_train)  # Calculamos la media y la varianza del conjutno de entrenamiento para la normalización

    ############### Implementamos el perceptrón multicapa ###############
    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

        # Vamos a ver los parches de una imagne simple
        plt.figure(figsize=(4, 4))
        image = x_train[np.random.choice(range(x_train.shape[0]))]
        plt.imshow(image.astype("uint8"))
        plt.axis("off")

        resized_image = tf.image.resize(
            tf.convert_to_tensor([image]), size=(image_size, image_size)
        )
        patches = Patches(patch_size)(resized_image)
        print(f"Image size: {image_size} X {image_size}")
        print(f"Patch size: {patch_size} X {patch_size}")
        print(f"Patches per image: {patches.shape[1]}")
        print(f"Elements per patch: {patches.shape[-1]}")

        n = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(n, n, i + 1)
            patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis

    ############### Contruimos el modelo ViT ###############
    def create_vit_classifier(self):
        print("Vamos a crear el clasificador...")
        training = Training()
        inputs = layers.Input(shape=training.input_shape)
        # Aumentamos los datos
        augmented = training.data_augmentation(inputs)
        # Creamos los parches
        patches = Patches(training.patch_size)(augmented)
        # Codificamos los parches
        enconded_patches = PatchEncoder(training.num_patches, training.projection_dim)(patches)

        # Creamos múltiples capas para el bloque del transformador
        for _ in range(training.transformer_layers):
            # Capa 1 de normalización
            x1 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
            # Creamos una capa de atención con múltiples cabeeras
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=training.num_heads, key_dim=training.projection_dim, dropout=0.1
            )(x1, x1)
            # Saltamos la primera conexión
            x2 = layers.Add()([attention_output, enconded_patches])
            # Segunda capa de normalización
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            training.mlp(x3, hidden_units=training.transformer_units, dropout_rate=0.1)
            # Saltamos la segunda conexión
            enconded_patches = layers.Add()([x3, x2])

        # Creamos un tensor con el batch_size y la projection_dim pertinentes
        representation = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Añadimos la MLP
        features = training.mlp(representation, hidden_units=training.mlp_head_units, dropout_rate=0.5)
        # Clasificamos la salida
        logits = layers.Dense(training.num_classes)(features)
        # Creamos el modelo de Keras.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def run_experiment(self, model):
        optimizer = tfa.optimizers.AdamW(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ],
        )

        checkpoint_filepath = "../tmp/checkpoint"
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        print("Entrenamos...")
        history = model.fit(
            x=self.x_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        model.save("../model/myModel.h5")
        model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = model.evaluate(self.x_test, self.y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        print("Hemos acabado el entrenamiento.")
        return history


############### Implementamos la creación del parche como una capa ###############
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


############### Implementamos la capa de codificación de los parches ###############
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# Compiamos, entrenamos y evaluamos el modelo

training = Training()
vit_classifier = Training.create_vit_classifier(training)
history = Training.run_experiment(training, vit_classifier)
# vit_classifier = create_vit_classifier()
# history = run_experiment(vit_classifier)

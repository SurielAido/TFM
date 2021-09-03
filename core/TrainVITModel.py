import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from utils.loading_dataset import LoadingDataset

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class TrainVITModel:
    def __init__(self, input_shape, learning_rate, weight_decay, batch_size, num_epochs, image_size, patch_size,
                 projection_dim, num_heads, transformer_layers, mlp_head_units,
                 destination_model_path='model/vit_model/vit_model.h5', load_dataset=False):
        self.input_shape = input_shape
        if load_dataset is False:
            self.num_classes = None
            self.x_train = None
            self.y_train = None
            self.x_test = None
            self.y_test = None
        else:
            dataset = LoadingDataset.take_me_dataset(
                LoadingDataset(input_shape[0], input_shape[1]))
            dataset = list(dataset)
            self.num_classes = dataset[0][4]
            self.x_train = dataset[0][0]
            self.y_train = dataset[0][2]
            self.x_test = dataset[0][1]
            self.y_test = dataset[0][3]
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.transformer_units = [
            projection_dim * 2,
            projection_dim
        ]
        self.transformer_layers = transformer_layers
        self.mlp_head_units = mlp_head_units
        self.destination_model_path = destination_model_path

    def data_augmentation(self):
        return keras.Sequential(
            [
                layers.experimental.preprocessing.Normalization(),
                layers.experimental.preprocessing.Resizing(self.image_size, self.image_size),
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.02),
                layers.experimental.preprocessing.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )

    def mlp(self, x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    def create_vit_classifier(self):
        print("Vamos a crear el clasificador...")
        inputs = layers.Input(shape=self.input_shape)
        # Aumentamos los datos
        augmented = self.data_augmentation()(inputs)
        # Creamos los parches
        patches = Patches(self.patch_size)(augmented)
        # Codificamos los parches
        enconded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)

        # Creamos múltiples capas para el bloque del transformador
        for _ in range(self.transformer_layers):
            # Capa 1 de normalización
            x1 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
            # Creamos una capa de atención con múltiples cabeeras
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
            )(x1, x1)
            # Saltamos la primera conexión
            x2 = layers.Add()([attention_output, enconded_patches])
            # Segunda capa de normalización
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP
            self.mlp(x3, hidden_units=self.transformer_units, dropout_rate=0.1)
            # Saltamos la segunda conexión
            enconded_patches = layers.Add()([x3, x2])
            
        # Creamos un tensor con el batch_size y la projection_dim pertinentes
        representation = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Añadimos la MLP
        features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.5)
        # Clasificamos la salida
        logits = layers.Dense(self.num_classes)(features)
        # Creamos el modelo de Keras.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def run_experiment(self):
        model = self.create_vit_classifier()
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

        checkpoint_filepath = "model/"
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

        model.load_weights(checkpoint_filepath)
        _, accuracy, top_5_accuracy = model.evaluate(self.x_test, self.y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        print("Hemos acabado el entrenamiento.")
        keras.models.save_model(model, self.destination_model_path)
        print("El modelo ha sido guardado, y puede encontrarse en ", self.destination_model_path)
        return history


############### Implementamos la creación del parche como una capa ###############
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config

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
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection_dim': self.projection_dim,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# vitModel = TrainVITModel((72, 72, 3), 0.001, 0.0001, 32, 100, 72, 6, 64, 4, 8, [2048, 1024], load_dataset=True)
# vitModel.run_experiment()

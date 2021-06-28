import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from utils.loading_dataset import LoadingDataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

############### Preparamos los datos ###############
input_shape = (112, 112, 3)

############### Preparamos los datos nuestros ###############
x_train, x_test, y_train, y_test, num_classes = LoadingDataset.take_me_dataset(
    LoadingDataset(input_shape[0], input_shape[1]))
print('Anchura: ', str(input_shape[0]))
print('Altura: ', str(input_shape[1]))
print('Número de clases: ', str(num_classes))

############### Configuramos los hiperparámetros ###############
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 10
image_size = 72  # Vamos a redimensionar las imágenes de entrada a este tamaño
patch_size = 6  # Tamaño de los "parches" que vamos a extraer de las imágenes de entrada
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim
]  # Tamaño de las capas del transformer
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Tamaó de las capas densas del clasificador final
destination_model_path = 'model/model.h5'

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
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


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


############### Contruimos el modelo ViT ###############
def create_vit_classifier():
    print("Vamos a crear el clasificador...")
    inputs = layers.Input(shape=input_shape)
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
        attention_output = tf.keras.layers.MultiHeadAttention(
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

        x4 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        # Creamos una capa de atención con múltiples cabeeras
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x4, x4)
        # Saltamos la primera conexión
        x5 = layers.Add()([attention_output, enconded_patches])
        # Segunda capa de normalización
        x6 = layers.LayerNormalization(epsilon=1e-6)(x5)
        # MLP
        mlp(x6, hidden_units=transformer_units, dropout_rate=0.1)
        # Saltamos la segunda conexión
        enconded_patches = layers.Add()([x6, x5])

        x7 = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
        # Creamos una capa de atención con múltiples cabeeras
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x7, x7)
        # Saltamos la primera conexión
        x8 = layers.Add()([attention_output, enconded_patches])
        # Segunda capa de normalización
        x9 = layers.LayerNormalization(epsilon=1e-6)(x8)
        # MLP
        mlp(x9, hidden_units=transformer_units, dropout_rate=0.1)
        # Saltamos la segunda conexión
        enconded_patches = layers.Add()([x9, x8])

    # Creamos un tensor con el batch_size y la projection_dim pertinentes
    representation = layers.LayerNormalization(epsilon=1e-6)(enconded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Añadimos la MLP
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Clasificamos la salida
    logits = layers.Dense(num_classes)(features)
    # Creamos el modelo de Keras.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# Compiamos, entrenamos y evaluamos el modelo
def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
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
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    print("Hemos acabado el entrenamiento.")
    keras.models.save_model(model, destination_model_path)
    print("El modelo ha sido guardado, y puede encontrarse en ", destination_model_path)
    return history

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

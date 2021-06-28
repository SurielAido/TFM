from tensorflow.keras.models import Model
from tensorflow.keras import models
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50
from utils.loading_dataset import LoadingDataset
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TransferLearningAboutPaper:

    def __init__(self, width_image, height_image, dimensions, epochs, batch_size, charge_dataset,
                 x_train=None, y_train=None, x_test=None, y_test=None, num_classes=51, pretrained_model='DEFAULT'):
        self.charge_dataset = charge_dataset
        if charge_dataset:
            dataset = self.take_me_my_dataset(width_image, height_image)
            print("Comenzamos con el proceso de transfer learning")
        else:
            dataset = None
        self.pretrained_model = pretrained_model
        self.width_image = width_image
        self.height_image = height_image
        self.dimensions = dimensions
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = pretrained_model
        if dataset is None:
            self.num_classes = num_classes
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test
        else:
            self.num_classes = dataset[4]
            self.x_train = dataset[0]
            self.y_train = dataset[2]
            self.x_test = dataset[1]
            self.y_test = dataset[3]

    def take_me_my_dataset(self, width_image, height_image):
        x_train, x_test, y_train, y_test, num_classes = LoadingDataset.take_me_dataset(
            LoadingDataset(width_image, height_image))
        return x_train, x_test, y_train, y_test, num_classes

    def generate_model(self):
        print("Vamos a generar el modelo")
        image_input = Input(shape=(self.width_image, self.height_image, self.dimensions))
        if self.pretrained_model == 'resnet':
            model = ResNet50(input_tensor=image_input, include_top=True, weights='imagenet')
            print("Vamos a usar ResNet50")
        elif self.pretrained_model == 'vgg':
            model = VGG16(input_tensor=image_input, include_top=True, weights='imagenet')
            print("Vamos a usar VGG16")
        else:
            model = InceptionV3(input_tensor=image_input, include_top=True, weights='imagenet')
            print("Vamos a usar InceptionV3")
        for layer in model.layers[:-1]:
            layer.trainable = False
        last_layer = model.layers[-1].output
        x = Dropout(0.5)(last_layer)
        x = Dense(224, activation='relu')(x)
        out = Dense(self.num_classes, activation="softmax")(x)
        final_model = Model(image_input, out)

        print("Compilamos el modelo")
        final_model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        if self.pretrained_model == 'resnet':
            destination_model_before_train_path = 'model/transfer_learning/transfer_resnet.h5'
        elif self.pretrained_model == 'vgg':
            destination_model_before_train_path = 'model/transfer_learning/transfer_vgg.h5'
        else:
            destination_model_before_train_path = 'model/transfer_learning_about_paper/transfer_paper_model.h5'
        models.save_model(final_model, destination_model_before_train_path)
        return final_model

    def train_model(self):
        print("Vamos a comenzar con el entrenamiento")
        model = models.load_model('model/transfer_learning_about_paper/trained_transfer_paper_model.h5')
        if model is not None:
            transfered_model = model
        else:
            transfered_model = self.generate_model()

        checkpoint_filepath = "model/transfer_learning_about_paper/checkpoint.h5"
        checkpoint_callback = callbacks.ModelCheckpoint(
            checkpoint_filepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )

        transfered_model.fit(
            x=self.x_train,
            # y=to_categorical(self.y_train, 3),
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.1,
            callbacks=[checkpoint_callback],
        )

        transfered_model.load_weights(checkpoint_filepath)
        # _, accuracy, top_5_accuracy = transfered_model.evaluate(self.x_test, self.y_test)
        # print(f"Test accuracy: {round(accuracy * 100, 2)}%")
        # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

        if self.pretrained_model == 'resnet':
            destination_model_trained_path = 'model/transfer_learning/trained_transfer_resnet.h5'
        elif self.pretrained_model == 'vgg':
            destination_model_trained_path = 'model/transfer_learning/trained_transfer_vgg.h5'
        else:
            destination_model_trained_path = 'model/transfer_learning_about_paper/trained_transfer_paper_model.h5'
        models.save_model(transfered_model, destination_model_trained_path)

        print("Hemos acabado el entrenamiento. El modelo está entrenado y guardado en", destination_model_trained_path)


# transfer = TransferLearningAboutPaper(112, 112, 3, 500, 32, True)
# if transfer.x_train is not None:
#     x_train = transfer.x_train
# if transfer.x_test is not None:
#     x_test = transfer.x_test
# if transfer.y_train is not None:
#     y_train = transfer.y_train
# if transfer.y_test is not None:
#     y_test = transfer.y_test
# if transfer.num_classes is not None:
#     num_classes = transfer.num_classes
# transfer.train_model()
# print("Vamos a vgg")
# transfer_vgg = TransferLearningAboutPaper(112, 112, 3, 500, 32,
#                                             False, x_train, y_train, x_test, y_test, num_classes, 'vgg')
# transfer_vgg.train_model()
# print("Vamos a resnet")
# transfer_resnet = TransferLearningAboutPaper(112, 112, 3, 500, 32,
#                                            False, x_train, y_train, x_test, y_test, num_classes, 'resnet')
# transfer_resnet.train_model()

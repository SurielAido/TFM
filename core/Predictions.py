import tensorflow as tf
from core.TrainVITModel import PatchEncoder, Patches
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.loading_dataset import LoadingDataset
from sklearn import metrics
from utils.utils import Utils
import numpy as np


class MyPredictions:

    def __init__(self, model_path, train_dir='dataset/train'):
        self.model_path = model_path
        self.train_dir = train_dir
        self.is_vit = False
        if 'transfer' in self.model_path:
            self.is_transfered = True
            self.width = 150
            self.height = 150
            if 'resnet' in self.model_path:
                self.model_base = ResNet50(weights='imagenet', include_top=False,
                                           input_shape=(self.width, self.height, 3))
                self.model_shapes = 5, 5, 2048
                self.base_type = 'resnet'
            elif 'vgg' in self.model_path:
                self.model_base = VGG16(weights='imagenet', include_top=False,
                                        input_shape=(self.width, self.height, 3))
                self.model_shapes = 4, 4, 512
                self.base_type = 'vgg'
            else:
                self.model_base = InceptionV3(weights='imagenet', include_top=False,
                                              input_shape=(self.width, self.height, 3))
                self.model_shapes = 3, 3, 2048
                self.base_type = 'inception'
        else:
            self.is_transfered = False
            self.width = 72
            self.height = 72
            if 'cnn' not in self.model_path:
                self.is_vit = True
            else:
                self.is_vit = False

    # './model/model.h5'
    def assing_model(self):
        if 'vit' in self.model_path:
            return tf.keras.models.load_model(self.model_path,
                                              custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
        else:
            return tf.keras.models.load_model(self.model_path)

    def make_a_prediction(self, img_path, model_trained=None):
        if model_trained is None:
            model_trained = self.assing_model()
        img = image.load_img(img_path, target_size=(self.width, self.height))
        img_2_array = image.img_to_array(img)
        if not self.is_transfered:
            img_batch = np.expand_dims(img_2_array, axis=0)
            # img_preprocessed = preprocess_input(img_batch)
            prediction = model_trained.predict(img_batch)
            res = Utils.class_2_string(np.argmax(prediction))

        else:
            features = self.model_base.predict(img_2_array.reshape(1, self.width, self.height, 3))

            try:
                prediction = model_trained.predict(features)
            except:
                prediction = model_trained.predict(
                    features.reshape(1, self.model_shapes[0] * self.model_shapes[1] * self.model_shapes[2]))

            result = np.argmax(np.array((prediction[0])))
            res = Utils.class_2_string(result)
        return res

    def run_experiment(self, model_trained=None):
        if model_trained is None:
            model_trained = self.assing_model()
        test_dir = 'dataset/train'
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_gen = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.width, self.height),
            color_mode="rgb",
            shuffle=False,
            class_mode='categorical',
            batch_size=1)

        filenames = test_gen.filenames
        nb_samples = len(filenames)

        if self.is_transfered:
            preds = self.model_base.predict_generator(test_gen, steps=nb_samples)
            preds = model_trained.predict(preds)
        elif self.is_vit:
            return self.predict_vit(model_trained)
        else:
            preds = model_trained.predict_generator(test_gen, steps=nb_samples)

        test_file_names = test_gen.filenames  # sequential list of name of test files of each sample
        test_labels = test_gen.labels  # is a sequential list  of test labels for each image sample
        class_dict = test_gen.class_indices  # a dictionary where key is the class name and value is the corresponding label for the class
        new_dict = {}
        for key in class_dict:  # set key in new_dict to value in class_dict and value in new_dict to key in class_dict
            value = class_dict[key]
            new_dict[value] = key

        k = 0
        for i, p in enumerate(preds):
            if k % 5000 == 0:
                print("Sigo funcionando")
            pred_index = np.argmax(p)  # get the index that has the highest probability
            pred_class = new_dict[pred_index]  # find the predicted class based on the index
            true_class = new_dict[test_labels[i]]  # use the test label to get the true class of the test file
            file = test_file_names[i]
            res = true_class + '#####' + file + '#####' + pred_class
            if self.is_transfered:
                if self.base_type == 'resnet':
                    Utils.write_file('predictions_files/resnet_predictions.txt', res)
                elif self.base_type == 'vgg':
                    Utils.write_file('predictions_files/vgg_predictions.txt', res)
                else:
                    Utils.write_file('predictions_files/inception_predictions.txt', res)
            else:
                if 'cnn' in self.model_path:
                    Utils.write_file('predictions_files/cnn_predictions.txt', res)
                else:
                    Utils.write_file('predictions_files/vit_predictions.txt', res)
            k += 1
        print("Iteraciones totales: " + str(k))
        return res

    def predict_vit(self, model):
        dataset = LoadingDataset.take_me_dataset(
            LoadingDataset(self.width, self.width), for_metrics=True
        )
        dataset = list(dataset)
        x_test = dataset[0][0]
        y_test = dataset[0][1]
        del dataset

        print("Dataset cargado, ahora voy a cargar el modelo")
        print("Muy bien, he asignado el modelo. El modelo es: ", model)
        pred = model.predict(x_test).argmax(axis=1)
        clasRep = metrics.classification_report(y_test, pred)
        print(clasRep)
        filename = 'vit_predictions_metrics.txt'
        file = open(filename, 'w')
        file.write(str(clasRep))

        return clasRep


model_path1 = 'model/vit_model/vit_model.h5'
prediction = MyPredictions(model_path1)
prediction.run_experiment()
# model = prediction.assing_model()
# print(prediction.make_a_prediction('../core/dataset/train/action/888.jpg', model))

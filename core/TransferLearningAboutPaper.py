import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam


class TransferLearningAboutPaper:

    def __init__(self, width, height, batch_size, number_classes, epochs, train_dir='dataset/train',
                 base_type='inception'):
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.number_classes = number_classes
        self.epochs = epochs
        self.train_dir = train_dir
        if base_type != 'vgg' and base_type != 'resnet' and base_type != 'inception':
            base_type = 'inception'
        if base_type == 'resnet':
            self.model_base = ResNet50(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
            self.model_shapes = 5, 5, 2048
            self.save_route = 'model/transfer_learning/resnet'
        elif base_type == 'vgg':
            self.model_base = VGG16(weights='imagenet', include_top=False, input_shape=(self.width, self.height, 3))
            self.model_shapes = 4, 4, 512
            self.save_route = 'model/transfer_learning/vgg'
        else:
            self.model_base = InceptionV3(weights='imagenet', include_top=False,
                                          input_shape=(self.width, self.height, 3))
            self.model_shapes = 3, 3, 2048
            self.save_route = 'model/transfer_learning/inception'
        self.base_type = base_type

    def create_model(self):
        print("Hemos escogido el modelo \'" + str(self.base_type) + "\' se va a cargar dicho modelo")
        train_features, train_labels = self.extract_features(self.train_dir, 34657)
        model = Sequential()
        model.add((Dense(256, activation='relu', input_shape=(self.model_shapes))))
        model.add(Dropout(0.5))
        model.add(GlobalAveragePooling2D())
        model.add(Dense(51, activation='softmax'))
        return model, train_features, train_labels

    def extract_features(self, directory, sample_count):
        datagen = ImageDataGenerator(rescale=1. / 255)
        features = np.zeros(shape=(sample_count, self.model_shapes[0], self.model_shapes[1], self.model_shapes[2]))
        labels = np.zeros(shape=(sample_count, 51))
        generator = datagen.flow_from_directory(
            directory,
            target_size=(self.width, self.height),
            batch_size=self.batch_size,
            class_mode='categorical',
        )

        i = 0
        for inputs_batch, labels_batch in generator:
            features_batch = self.model_base.predict(inputs_batch)
            features[i * self.batch_size: (i + 1) * self.batch_size] = features_batch
            labels[i * self.batch_size: (i + 1) * self.batch_size] = labels_batch
            i += 1
            if i * self.batch_size >= sample_count:
                break
        return features, labels

    def train_model(self):
        model, train_features, train_labels = self.create_model()

        checkpoint = ModelCheckpoint(self.save_route + '/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1,
                                     monitor='val_loss', save_best_only=True, mode='auto')
        model.compile(
            optimizer=Adam(),
            loss='categorical_crossentropy',
            metrics=['acc']
        )
        model.fit(
            train_features, train_labels,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[checkpoint],
            validation_split=0.25
        )
        save_model(model, self.save_route + 'model_' + self.base_type + '.h5')
        return model


# imagen = 'dataset/train/action/ddd.jpg'
# transfer_vgg = TransferLearningAboutPaper(150, 150, 32, 51, 200, base_type='vgg')
# transfer_vgg.train_model()
# transfer_inception = TransferLearningAboutPaper(150, 150, 32, 51, 200)
# transfer_inception.train_model()
# transfer_resnet = TransferLearningAboutPaper(150, 150, 32, 51, 200, base_type='resnet')
# transfer_resnet.train_model()

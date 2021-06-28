import numpy as np
from sklearn.model_selection import train_test_split
from utils.utils import Utils
import os, cv2


class LoadingDataset:

    def __init__(self, width, height):
        self.widht = width
        self.height = height

    # def __iter__(self, b):
    #     for a in b:
    #         yield a

    def create_dataset(self, img_folder):
        i = 1
        # img_data_array = np.array([])
        # class_name = []
        img_data_array = []
        class_name = []
        for dir, _, files in os.walk(img_folder):
            for file in files:
                if i % 5000 == 0:
                    print("Sigo funcionando. Voy por la iteración nº " + str(i))
                i += 1
                image_path = os.path.join(dir, file)
                image = cv2.imread(image_path, cv2.COLOR_BayerRG2RGB)
                image = cv2.resize(image, (self.height, self.widht), cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                # img_data_array = np.append(img_data_array, image)
                img_data_array.append(image)
                class_name.append(Utils.transform_path_for_dataset(dir))
        print("Número de iteraciones total: " + str(i))
        yield img_data_array, class_name

    def take_me_dataset(self, path=None, for_metrics=False):
        print("Comenzamos...")
        if path is not None:
            data = list(self.create_dataset(path))
            img_data, class_name = data[0], data[1]
            del data
        else:
            data = list(self.create_dataset('dataset/train'))
            img_data, class_name = data[0][0], data[0][1]
            del data
        target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
        target_val = [target_dict[class_name[i]] for i in range(len(class_name))]
        del class_name
        del target_dict
        classesSet = set(target_val)
        numClasses = len(classesSet)
        x = np.array(img_data, np.float32)
        del img_data
        y = np.array(list(map(int, target_val)))
        del target_val
        if for_metrics:
            yield x, y, numClasses
        else:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
            yield X_train, X_test, y_train, y_test, numClasses



# data = LoadingDataset(112, 112)
# data.save_dataset()

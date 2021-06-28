import os
import clip
import torch
import numpy as np
from PIL import Image
from utils.utils import Utils


# Load the model

class ClipEvaluating:

    def __init__(self, img_folder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        modeled = clip.load('ViT-B/32', self.device)
        self.model = modeled[0]
        self.preprocess = modeled[1]
        self.img_folder = img_folder
        self.class_name = self.give_me_my_classes(img_folder)

    @staticmethod
    def give_me_my_classes(self, img_folder):
        class_name = []
        for dir1 in os.listdir(img_folder):
            for _ in os.listdir(os.path.join(img_folder, dir1)):
                class_name.append(dir1)
            target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
            classes_array = []
            clases = target_dict.keys()
            for c in clases:
                classes_array.append(c)
        return classes_array

    # class_name = give_me_my_classes(r'D:/Suriel/Universidad de Sevilla/Máster en Lógica, Computación e Inteligencia Artificial/Trabajo de Fin de Máster/Desarrollo/TFM/core/dataset/train')

    def calculate_prediciton(self, image):
        # Prepare the inputs
        image = Image.open(image)
        class_id = 1
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in self.class_name]).to(self.device)

        # Calculate features
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(1)

        # Print the result
        result = []
        # print("\nClass predicted: ")
        for value, index in zip(values, indices):
            aux = f"{self.class_name[index]:>16s}: {100 * value.item():.2f}%"
            # print(aux)
            result.append(aux)
        return result[0].strip()

    def run_experiment(self):
        for dir, _, files in os.walk(self.img_folder):
            for file in files:
                image_path = os.path.join(dir, file)
                # image_path_safe = image_path.lower()
                str_path_2_write = Utils.format_path(image_path)
                # self.write_file('clip_predictions.txt', str_path_2_write[1])
                image_2_class_prediction = str_path_2_write[0] + '#####' + self.calculate_prediciton(image_path)
                text2write = str_path_2_write[1] + '#####' + image_2_class_prediction
                print(text2write)
                Utils.write_file('predictions_files/clip_predictions.txt', text2write)
                # prediction.run(image_path_safe)

    def extract_accuracy(self):
        file = open('predictions_files/clip_predictions.txt', 'r')
        lines = file.readlines()

        success = 0
        fails = 0
        for line in lines:
            splitted = line.split('#####')
            real_class = splitted[0]
            # film_name = splitted[1]
            predicted_class = splitted[2].split(':')[0]

            if real_class == predicted_class:
                success += 1
            else:
                fails += 1
        success_and_fails_message = 'Exitos: ' + str(success) + ' / Fallos: ' + str(fails)
        print(success_and_fails_message)
        self.write_file('predictions_files/clip_predictions.txt', success_and_fails_message)
        return success, fails


# for dir, _, files in os.walk('validation/images_test'):


# clipModel = ClipEvaluating('../core/dataset/train')
# clipModel.run_experiment()
# clipModel.extract_metrics()
# print(clipModel.format_path('../core/dataset/train\\Abduction-hostage\\download (1).jpg'))

from PIL import Image
import os


class Jpeg2JpgConversor:
    def __init__(self, path):
        self.path = path

    def converter(self, mode='jpeg', target='jpg'):
        i = 1
        if mode != 'jpg' and mode != 'jpeg' and mode != 'png':
            mode = 'invalid'
        if target != 'jpg' and target != 'jpeg' and target != 'png':
            target = 'invalid'
        if mode == 'invalid' or target == 'invalid':
            print("No se ha podido realizar la conversión. Los modos admitidos son: \'jpeg, \' jpg "
                  "y \'png.\nPor favor, escríbalos correctamente si quiere continuar.")
            return False
        if mode == target:
            print("Está intentando convertir la imagen al mismo formato en el que se encuentra. La aplicación no"
                  "realizará ninguna acción")
        else:
            for dir, _, files in os.walk(self.path):
                for file in files:
                    image_path = os.path.join(dir, file)
                    image_path_safe = image_path.lower()
                    if self.give_me_image_type(image_path_safe) == mode:
                        print("Imagen: " + image_path_safe)
                        # importing the image
                        im = Image.open(image_path_safe)
                        # converting to jpg
                        rgb_im = im.convert("RGB")
                        # exporting the image
                        rgb_im.save(self.change_my_image_type(image_path_safe, target))
                        # Deleting the original image
                        os.remove(image_path)
                        print("Imagen número " + str(i) + " convertida")
                        i += 1

    def give_me_image_type(self, path):
        path_array = str(path).split('.')
        if path_array[len(path_array)-1] == 'jpeg':
            return 'jpeg'
        elif path_array[len(path_array)-1] == 'jpg':
            return 'jpg'
        elif path_array[len(path_array)-1] == 'png':
            return 'png'
        else:
            return 'invalid'

    @staticmethod
    def change_my_image_type(path, target):
        path_array = str(path).split('.')
        if path_array[len(path_array)-1] != 'jpg' and path_array[len(path_array)-1] != 'jpeg' and path_array[len(path_array)-1] != 'png':
            return 'invalid'
        if path_array[len(path_array)-1] == target:
            return None
        else:
            return str(path).replace(path_array[len(path_array)-1], target)


# conversor = Jpeg2JpgConversor(r'./dataset/train/adventure')
# conversor = Jpeg2JpgConversor(r'../core/dataset/train')
# Jpeg2JpgConversor.conversor(conversor, 'jpeg', 'jpg')
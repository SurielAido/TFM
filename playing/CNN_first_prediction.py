import numpy
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow import keras as k

longitud, altura = 100, 100 # Tiene que ser la misma que la del modelo, claro
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn = k.models.load_model(modelo)
cnn.load_weights(pesos)

def predict(file):
    imagen_entrada = load_img(file, target_size=(longitud, altura))
    imagen_entrada = img_to_array(imagen_entrada)   # pasa la imagen a array
    imagen_entrada = numpy.expand_dims(imagen_entrada, 0)   # Padding con 0
    prediccion = cnn.predict(imagen_entrada)   # nos va a traer un 1 donde crea que es correcta ([[1,0,0]], p.e.)
    resultado = prediccion[0]   # La predicción nos devuelve un array de 2d, y queremos que nos traiga solo la primera, que es la que tiene el array [0,0,1]
    respuesta = numpy.argmax(resultado) # Si tenemos [0, 1, 0], nos va a devolver 1, ya que es la posición del array donde está el valor más alto (el 1).
    if respuesta == 0:
        print('Perro')
    elif respuesta == 1:
        print('Gato')
    elif respuesta == 2:
        print('Gorila')
    return respuesta

predict('nano.jpeg')   #Le metemos en la raíz una imagen y le metemos el nombre a la función.
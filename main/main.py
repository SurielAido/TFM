from core.ClipPredictions import ClipEvaluating
from core.Metrics import MyMetrics
from core.MyConvolutionalNetwork import MyConvolutionalNetwork
from core.Predictions import MyPredictions
from core.TrainVITModel import TrainVITModel
from core.TransferLearningAboutPaper import TransferLearningAboutPaper


def executeTFM(action):
    vit, cnn, resnet, vgg, inception = charge_classes_training_execution()
    print("Va a ejecutarse el TFM de Suriel Aido Teruel. A continuación se mostrarán en las próximas líneas"
          " las instrucciones que van a ejecutarse. Esta ejecución se va a intetar minimizar, de forma que"
          " algunas de las opciones estarán al mínimo. Esto implica que, por ejemplo, los entrenamientos se"
          " harán por defecto con una sola época, para comprobar la ejecución. Y se guardarán todos los resultados"
          " en una carpeta llamada /ejecucion (dentro de esta misma carpeta, \"main\"")
    print("...")
    print("Vamos a entrenar, en primer lugar, el entrenamiento del modelo ViT (Visual Transformer")
    vit.run_experiment()
    print("...")
    print("Ahora, vamos a realizar el entrenamiento de la red convolucional tradicional")
    cnn.train_model()
    print("...")
    print("Ahora, vamos a realizar el entrenamiento con el uso de transfer learning con InceptionV3 como modelo base")
    inception.train_model()
    print("...")
    print("Ahora, vamos a realizar el entrenamiento con el uso de transfer learning con VGG16 como modelo base")
    vgg.train_model()
    print("...")
    print("Ahora, vamos a realizar el entrenamiento con el uso de transfer learning con ResNet50 como modelo base")
    resnet.train_model()

    print("...")
    print("...")
    print("...")

    vit_pred, cnn_pred, inception_pred, vgg_pred, resnet_pred, clip_pred = charge_predictions_execution()
    print(
        "Tras haber entrenado los modelo, vamos a hacer las predicciones pertinentes. Dichas predicciones se almacenarán"
        "en /ejecucion/train_files/nombre_modelo.txt")
    print("En primer lugar, el modelo VIT")
    vit_pred.run_experiment()
    print("A continuación, el modelo CNN")
    cnn_pred.run_experiment()
    print("A continuación, el modelo entrenado con transfer learning basado en InceptionV3")
    inception_pred.run_experiment()
    print("A continuación, el modelo entrenado con transfer learning basado en VGG16")
    vgg_pred.run_experiment()
    print("A continuación, el modelo entrenado con transfer learning basado en ResNet50")
    resnet_pred.run_experiment()
    print("por úlitmo, el modelo CLIP")
    MyPredictions('predictions_files/clip_predictions.txt').run_experiment()

    print("...")
    print("...")
    print("...")

    metrics_vit, metrics_cnn, metrics_inception, metrics_vgg, metrics_resnet, metrics_clips = charge_metrics_execution()
    print(
        "Tras haber hecho las predicciones, vamos a sacar las métricas de cada uno de los modelos "
        "en base a las predicciones realizadas")
    print("En primer lugar, el modelo VIT")
    metrics_vit.calculate_matrix()
    print("A continuación, el modelo CNN")
    metrics_cnn.calculate_matrix()
    print("A continuación, el modelo entrenado con transfer learning basado en InceptionV3")
    metrics_inception.calculate_matrix()
    print("A continuación, el modelo entrenado con transfer learning basado en VGG16")
    metrics_vgg.calculate_matrix()
    print("A continuación, el modelo entrenado con transfer learning basado en ResNet50")
    metrics_resnet.calculate_matrix()
    print("por úlitmo, el modelo CLIP")
    metrics_clips.calculate_matrix()


def charge_classes_training_execution():
    vit = TrainVITModel((72, 72, 3), 0.001, 0.0001, 32, 100, 72, 6, 64, 4, 8, [2048, 1024], load_dataset=True)
    # CNN está mal, tengo que mirar los parámetros.
    cnn = MyConvolutionalNetwork(100, 112, 112, 10, 500, 100, 32, 64, (3, 3), (2, 2), (2, 2), 51, 0.0005)
    transfer_resnet = TransferLearningAboutPaper(150, 150, 32, 51, 200, base_type='resnet')
    transfer_vgg = TransferLearningAboutPaper(150, 150, 32, 51, 200, base_type='vgg')
    transfer_inception = TransferLearningAboutPaper(150, 150, 32, 51, 200)
    return vit, cnn, transfer_resnet, transfer_vgg, transfer_inception


def charge_predictions_execution():
    vit_predictions = MyPredictions('execution/predictions_files/vit_predictions.txt')
    cnn_predictions = MyPredictions('execution/predictions_files/cnn_predictions.txt')
    inception_predictions = MyPredictions('execution/predictions_files/inception_predictions.txt')
    vgg_predictions = MyPredictions('execution/predictions_files/vgg_predictions.txt')
    resnet_predictions = MyPredictions('execution/predictions_files/resnet_predictions.txt')
    clip_predictions = MyPredictions('execution/predictions_files/clip_predictions.txt')

    return vit_predictions, cnn_predictions, inception_predictions, vgg_predictions, resnet_predictions, clip_predictions


def charge_metrics_execution():
    metrics_clips = MyMetrics('execution/predictions_files/clip_predictions.txt')
    metrics_cnn = MyMetrics('execution/predictions_files/cnn_predictions.txt')
    metrics_inception = MyMetrics('execution/predictions_files/inception_predictions.txt')
    metrics_vgg = MyMetrics('execution/predictions_files/vgg_predictions.txt')
    metrics_resnet = MyMetrics('execution/predictions_files/clip_predictions.txt')
    metrics_vit = MyPredictions('execution/predictions_files/vit_predictions.txt')

    return metrics_vit, metrics_cnn, metrics_inception, metrics_vgg, metrics_resnet, metrics_clips

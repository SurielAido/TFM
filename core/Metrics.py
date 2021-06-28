from sklearn import metrics
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)


class MyMetrics:

	def __init__(self, file):
		self.file = file

	def calculate_matrix(self):
		f = open(self.file, 'r')
		lines = f.readlines()
		y_true = []
		y_pred = []
		for line in lines:
			splitted = line.split('#####')
			if 'clip' in self.file:
				if line == lines[-1]:
					continue
				y_pred.append(splitted[2].split(':')[0])
			else:
				y_pred.append(splitted[2].rstrip("\n"))
			y_true.append(splitted[0])
		conMax = metrics.confusion_matrix(y_true, y_pred)
		clasRep = metrics.classification_report(y_true, y_pred, digits=4)

		filename = 'metrics/' + self.file.split('/')[-1].split('.')[0] + '_metrics.txt'
		file = open(filename, 'w')
		file.write(str(clasRep))
		return conMax, clasRep


metrics_clips = MyMetrics('predictions_files/clip_predictions.txt')
metrics_cnn = MyMetrics('predictions_files/cnn_predictions.txt')
metrics_inception = MyMetrics('predictions_files/inception_predictions.txt')
metrics_vgg = MyMetrics('predictions_files/vgg_predictions.txt')
metrics_resnet = MyMetrics('predictions_files/resnet_predictions.txt')
# metrics_vit = MyMetrics('predictions_files/vit_predictions.txt')

metrics_clips.calculate_matrix()
metrics_cnn.calculate_matrix()
metrics_inception.calculate_matrix()
metrics_vgg.calculate_matrix()
metrics_resnet.calculate_matrix()
# metrics_vit.calculate_matrix()
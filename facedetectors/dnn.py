import os
from PIL import Image
import numpy as np
import cv2
from .face_detector import face_detector
import unittest

class DNN(face_detector):
	def __init__(self, model_path, input_details = (300, 300), score_threshold = 0.6):
		self.model = model_path
		directory = os.path.dirname(__file__)
		self.classifier = cv2.dnn
		self.configfile = os.path.join(directory, "model_files/deploy.prototxt.txt")
		self.net = self.classifier.readNetFromCaffe(self.configfile, os.path.join(directory, model_path))
		super().__init__(input_details, score_threshold)
	
	def prepare_image_path(self, image_path):
		img = Image.open(image_path)
		return self.prepare_image(img), img
	
	def prepare_image(self, img):
		return super().prepare_image(img)
	
	def draw_boxes_and_save(self, boxes, labels, scores, img, output_path):
		super().draw_boxes_and_save(boxes, labels, scores, img, output_path)
	
	def reformat_face(self, x, y, x1, y1):
		w = int(x1-x)
		h = int(y1-y)
		return (x,y,w,h)
	
	def setInputSize(self, input_details):
		self.width, self.height = input_details
	
	def predict(self, image):
		result = []
		#processed_image = prepare_image(image)
		processed_image = cv2.resize(image, (300, 300))
		blob = self.classifier.blobFromImage(processed_image, 
			1.0,
			(300, 300),
			(104.0, 117.0, 123.0))
		self.net.setInput(blob)	
		faces = self.net.forward()
		for i in range(faces.shape[2]):
			confidence = faces[0, 0, i, 2]
			if confidence > self.score_threshold:
				box = faces[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
				(x,y,x1,y1) = box.astype("int")
				result.append(self.reformat_face(x, y, x1, y1))
		return result

class Test_DNN(unittest.TestCase):
	def create_dnn(self):
		test_detector = DNN("model_files/res10_300x300_ssd_iter_140000.caffemodel", (4,4), 0.8)
		self.assertEqual(test_detector.height, 4)
		self.assertEqual(test_detector.weight, 4)
		self.assertEqual(test_detector.score_threshold, 0.8)

	def create_test_detector_default(self):
		test_detector = DNN("model_files/res10_300x300_ssd_iter_140000.caffemodel")
		self.assertEqual(test_detector.height, 300)
		self.assertEqual(test_detector.weight, 300)
		self.assertEqual(test_detector.score_threshold, 0.6)

	def test_reformat(self):
		test_detector = DNN("model_files/res10_300x300_ssd_iter_140000.caffemodel")
		(x, y, w, h) = test_detector.reformat_face(10, 10, 100, 100)
		self.assertEqual(w, 90)
		self.assertEqual(h, 90)

	def test_prediction(self):
		test_detector = DNN("model_files/res10_300x300_ssd_iter_140000.caffemodel")
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img2 = np.asarray(img, dtype=np.float32)
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

		test_detector.setInputSize((1912, 1054))

		result = test_detector.predict(img2)
		list_len = len(result)

		label = []
		score = []
		for i in range(list_len):
			label.append("box " + str(i))
			score.append(0.5)

		print(result)
		print(score)

if __name__=='__main__':
	unittest.main()
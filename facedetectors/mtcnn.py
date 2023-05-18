import os
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from .face_detector import face_detector
import unittest
import cv2

class MTCNNDet(face_detector):
	def __init__(self, input_details, score_threshold = 0.6):
		self.classifier = MTCNN()
		super().__init__(input_details, score_threshold)

	def prepare_image_path(self, image_path):
		img = Image.open(image_path)
		return self.prepare_image(img), img

	def prepare_image(self, img):
		return super().prepare_image(img)

	def draw_boxes_and_save(self, boxes, labels, scores, img, output_path):
		super().draw_boxes_and_save(boxes, labels, scores, img, output_path)

	def setInputSize(self, input_details):
		self.width, self.height = input_details

	def predict(self, image):
		results = []
		#processed_image = prepare_image(image)
		processed_image = image
		faces = self.classifier.detect_faces(image)
		for face in faces:
			results.append(face['box'])
		return results

class Test_MTCNN(unittest.TestCase):
	def create_dnn(self):
		test_detector = MTCNNDet((4,4), 0.8)
		self.assertEqual(test_detector.height, 4)
		self.assertEqual(test_detector.weight, 4)
		self.assertEqual(test_detector.score_threshold, 0.8)

	def create_test_detector_default(self):
		test_detector = MTCNNDet((200,200))
		self.assertEqual(test_detector.score_threshold, 0.6)

	def test_prediction(self):
		test_detector = MTCNNDet((200,200))
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
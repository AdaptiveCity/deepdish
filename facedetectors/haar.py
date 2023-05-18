import os
from PIL import Image
import numpy as np
import cv2
from .face_detector import face_detector
import unittest

def calculate_iou(box1, box2):
	box1 = [float(x) for x in box1]
	box2 = [float(x2) for x2 in box2]

	(x1, y1, w1, h1) = box1
	(x2, y2, w2, h2) = box2

	(x01, y01, x02, y02) = (x1, y1, x2, y2)
	(x11, y11, x12, y12) = (x1+w1, y1+h1, x2+w2, y2+h2)

	final_x0 = max(x01, x02)
	final_y0 = max(y01, y02)

	final_x1 = max(x11, x12)
	final_y1 = max(y11, y12)
	#print(final_x0, final_y0, final_x1, final_y1)

	if(final_x1 - final_x0 <= 0) or (final_y1 - final_y0 <= 0):
		return 0.0

	intersection = (final_x1 - final_x0) * (final_y1 - final_y0)

	union1 = (x11 - x01) * (y11 - y01)
	union2 = (x12 - x02) * (y12 - y02)
	union = union1 + union2 - intersection

	return intersection / union

class HaarCascades(face_detector):
	def __init__(self, model_path, model_path2, input_details, score_threshold = 0.6):
		self.model = model_path
		directory = os.path.dirname(__file__)
		self.classifier = cv2.CascadeClassifier(os.path.join(directory, model_path))
		self.classifier2 = cv2.CascadeClassifier(os.path.join(directory, model_path2))
		super().__init__(input_details, score_threshold)

	def prepare_image_path(self, image_path):
		img = Image.open(image_path)
		return self.prepare_image(img), img

	def prepare_image(self, img):
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		return gray_img

	def draw_boxes_and_save(self, boxes, labels, scores, img, output_path):
		super().draw_boxes_and_save(boxes, labels, scores, img, output_path)
	
	def reformat_face(self, x, y, x1, y1):
		w = int(x1-x)
		h = int(y1-y)
		return (x,y,w,h)
	
	def setInputSize(self, input_details):
		self.width, self.height = input_details
		
	def predict(self, image):
		processed_image = self.prepare_image(image)

		faces = self.classifier.detectMultiScale3(processed_image,
			scaleFactor=1.1,
			minNeighbors=3,
			minSize=(30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE,
			outputRejectLevels = True
		)
		rects = faces[0]
		neighbours = faces[1]
		weights = faces[2]
		result = []
		
		for i in range(len(weights)):
			if(weights[i] >= self.score_threshold):
				#(x, y, x1, y1) = rects[i]
				#result.append(self.reformat_face(x,y,x1,y1))
				result.append(rects[i])

		faces2 = self.classifier2.detectMultiScale3(processed_image,
			scaleFactor=1.1,
			minNeighbors=3,
			minSize=(30, 30),
			flags = cv2.CASCADE_SCALE_IMAGE,
			outputRejectLevels = True
		)

		rects2 = faces2[0]
		neighbours2 = faces2[1]
		weights2 = faces2[2]

		for i in range(len(weights2)):
			if(weights2[i] >= self.score_threshold):
				#(x, y, x1, y1) = rects[i]
				#result.append(self.reformat_face(x,y,x1,y1))
				for box in result:
					if calculate_iou(box, rects2[i]) < 0.5:
						result.append(rects2[i])
		return result

class Test_Haar(unittest.TestCase):
	def create_dnn(self):
		test_detector = HaarCascades("model_files/haarcascade_frontalface2.xml","model_files/haarcascade_profileface.xml",  (4,4), 0.8)
		self.assertEqual(test_detector.height, 4)
		self.assertEqual(test_detector.weight, 4)
		self.assertEqual(test_detector.score_threshold, 0.8)

	def create_test_detector_default(self):
		test_detector = HaarCascades("model_files/haarcascade_frontalface2.xml", "model_files/haarcascade_profileface.xml", (200, 200))
		self.assertEqual(test_detector.score_threshold, 0.6)

	def test_reformat(self):
		test_detector = HaarCascades("model_files/haarcascade_frontalface2.xml","model_files/haarcascade_profileface.xml", (200, 200))
		(x, y, w, h) = test_detector.reformat_face(10, 10, 100, 100)
		self.assertEqual(w, 90)
		self.assertEqual(h, 90)

	def test_prediction(self):
		test_detector = HaarCascades("model_files/haarcascade_frontalface2.xml", "model_files/haarcascade_profileface.xml", (200, 200))
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img2 = np.asarray(img, dtype=np.uint8)
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
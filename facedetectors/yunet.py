import os
from PIL import Image
import numpy as np
import cv2
from .face_detector import face_detector
import unittest

class Yunet(face_detector):
	def __init__(self, model_path, input_details, score_threshold = 0.9, nms_threshold = 0.3, topK = 5000):
		self.model = model_path
		directory = os.path.dirname(__file__)
		self.classifier = cv2.FaceDetectorYN_create(os.path.join(directory, model_path), "", (0, 0))
		self.height, self.width = input_details
		self.classifier.setScoreThreshold(score_threshold)
		self.classifier.setNMSThreshold(nms_threshold)
		self.classifier.setTopK(topK)

	def prepare_image_path(self, image_path):
	    img = Image.open(image_path)
	    return self.prepare_image(img), img

	def prepare_image(self, img):
	    return super().prepare_image(img)

	def draw_boxes_and_save(self, boxes, labels, scores, img, output_path):
	    super().draw_boxes_and_save(boxes, labels, scores, img, output_path)

	def setInputSize(self, input_details):
		self.width, self.height = input_details
		self.classifier.setInputSize(input_details)

	def predict(self, image):
		#processed_image = self.prepare_image(image)
		#processed_image = image
		_, faces = self.classifier.detect(image)
		return faces

class Test_Yunet(unittest.TestCase):
    def create_yunet(self):
        test_detector = Yunet("model_files/face_detection_yunet_2022mar.onnx", (4,4), 0.8)
        self.assertEqual(test_detector.height, 4)
        self.assertEqual(test_detector.weight, 4)
        self.assertEqual(test_detector.score_threshold, 0.8)

    def create_test_detector_default(self):
        test_detector = Yunet("model_files/face_detection_yunet_2022mar.onnx", (200, 200))
        self.assertEqual(test_detector.score_threshold, 0.6)

    def test_prediction(self):
        test_detector = Yunet("model_files/face_detection_yunet_2022mar.onnx", (200, 200))
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
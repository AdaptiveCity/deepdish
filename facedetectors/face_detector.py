import os
from PIL import Image
from PIL import ImageDraw
import numpy as np
import unittest

class face_detector:
    def __init__(self, input_details, score_threshold=0.7):
        self.height, self.width = input_details
        self.score_threshold = score_threshold

    def prepare_image_path(self, image_path):
        img = Image.open(image_path)
        return self.prepare_image(img), img

    def prepare_image(self, img):
        img_resized = img.convert('RGB').resize((self.width, self.height), Image.ANTIALIAS)
        return img_resized

    def draw_boxes_and_save(self, boxes, labels, scores, img, output_path):
        self.draw_boxes(boxes, labels, scores, img)
        img.save(output_path)

    def draw_boxes(self, boxes, labels, scores, img):
        draw = ImageDraw.Draw(img)
        original_size = np.array(img.size)
        detection_size = np.array((self.width, self.height))
        color = tuple(np.random.randint(0, 256, 3))
        for box, score, lbl in zip(boxes, scores, labels):
            rbox=list(box)
            draw.rectangle(rbox, outline=color)
            draw.text(rbox[:2], '{} {:.2f}%'.format(lbl, score * 100), fill=color)

class Test_Face_Detector(unittest.TestCase):
    def create_test_detector(self):
        test_detector = face_detector((4,4), 0.8)
        self.assertEqual(test_detector.height, 4)
        self.assertEqual(test_detector.width, 4)
        self.assertEqual(test_detector.score_threshold, 0.8)

    def create_test_detector_default(self):
        test_detector = face_detector((4,4))
        self.assertEqual(test_detector.height, 4)
        self.assertEqual(test_detector.width, 4)
        self.assertEqual(test_detector.score_threshold, 0.7)

    def prepare_image_path(self):
        test_detector = face_detector((4,4))
        directory = os.path.dirname(__file__)

        img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))

        width, height = img1.size
        self.assertEqual(width, 4)
        self.assertEqual(height, 4)

    def test_draw_boxes(self):
        test_detector = face_detector((4,4))
        directory = os.path.dirname(__file__)

        img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))

        boxes = [[0, 30, 14, 15], [20, 50, 40, 30]]
        lables = ['box1', 'box2']
        scores = [0.5, 0.6]

        test_detector.draw_boxes_and_save(boxes, lables, scores, img, os.path.join(directory, 'test_images/sample_output_2.png'))

if __name__=='__main__':
	unittest.main()

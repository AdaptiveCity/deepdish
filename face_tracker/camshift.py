import numpy as np
import cv2
import unittest
from .bbox import bbox
from PIL import Image, ImageDraw
import os

class Camshift():

	def __init__(self, current_bbox, frame):
		self.bbox = current_bbox
		self.create_hist(frame)
		self.mask = None
		self.backprojection = None

		self.term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1)

	def create_hist(self, frame):
		x = self.bbox.x
		y = self.bbox.y
		w = self.bbox.w
		h = self.bbox.h

		temp_box = frame[y:y+h, x:x+w]
		hsv_box = cv2.cvtColor(temp_box, cv2.COLOR_BGR2HSV)

		lower_mask = np.array((0.0, 50.0, 50.0))
		upper_mask = np.array((180.0, 255.0, 255.0))

		new_mask = cv2.inRange(hsv_box, lower_mask, upper_mask)

		self.hist = cv2.calcHist([hsv_box], [0], new_mask, [180], [0,180])

		cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

	def get_backprojection(self, frame):
		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		new_backprojection = cv2.calcBackProject([frame_hsv], [0], self.hist, [0,180], 1)
		self.backprojection = new_backprojection

		return new_backprojection.copy()

	def compute_moment(frame, x, y):
		result = 0
		i = 0
		for row in frame:
			j = 0
			for cell in row:
				result += ((pow(i, x)) * (pow(j, y)) * cell)
				j += 1
			i += 1
		return result

	def mean_shift(box, x_c, y_c):
		x = abs(box.x + y_c - (box.w/2))
		y = abs(box.y + x_c - (box.h/2))

		result = bbox()
		result.set_values(x, y, w, h)
		return result

	def compute_bbox_alt(self, frame):
		count = 0
		m00 = 1

		while(m00 != 0):
			distr = self.get_backprojection(frame)
			m00 = compute_moment(distr, 0, 0)
			m10 = compute_moment(distr, 1, 0)
			m01 = compute_moment(distr, 0, 1)

			x_c = m10/m00
			y_c = m01/m00

			new_bbox = mean_shift(self.bbox, x_c, y_c)
			converged = abs(new_bbox.x - self.bbox.x) < 2 and abs(new_bbox.y - self.bbox.y) < 2
			if(converged or count > 20):
				h = int(math.sqrt(m00)) * 2
				w = sq*1.5

				self.bbox.set_values(new_bbox.x, new_bbox.y, w, h)
				m00 = 0
			self.bbox = new_bbox
			count += 1

	def compute_bbox(self, frame):
		self.get_backprojection(frame)

		x = self.bbox.x
		y = self.bbox.y
		w = self.bbox.w
		h = self.bbox.h
		box = (x, y, w, h)
		self.rotated, new_bbox = cv2.CamShift(self.backprojection, box, self.term_criteria)

		self.rotated = cv2.boxPoints(self.rotated)
		#print("ROTATED 1")
		#print(self.rotated)
		self.rotated = np.int0(self.rotated)
		#print("ROTATED 2")
		#print(self.rotated)
		(x2, y2, w2, h2) = new_bbox
		self.bbox = bbox()
		self.bbox.set_values(x2, y2, w2, h2)

	def get_bbox(self):
		return (self.bbox.x, self.bbox.y, self.bbox.w, self.bbox.h)

	def get_rotated(self):
		return self.rotated

class Test_camshift(unittest.TestCase):
	def test_create_hist(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

		cam_test = Camshift(test_box, img)

		cam_test.create_hist(img)

		self.assertEqual(len(cam_test.hist), 180)

	def test_get_backprojection(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

		cam_test = Camshift(test_box, img)

		back_proj = cam_test.get_backprojection(img)

		self.assertEqual(len(back_proj), 1054)

	def test_compute_bbox(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img)
		img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

		cam_test = Camshift(test_box, img)

		back_proj = cam_test.get_backprojection(img)

		cam_test.compute_bbox(img)

		print(cam_test.get_bbox)

if __name__=='__main__':
	unittest.main()
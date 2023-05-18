import cv2
import numpy as np
from PIL import Image, ImageDraw
from .bbox import bbox
import time
import os

import unittest

class optical_flow:
	def __init__(self, old_bbox):
		self.prev_bbox = old_bbox
		self.new_bbox = bbox()

		self.old_points = []
		self.new_points = []
		self.reverse_points = []

		self.tracking = False
		term_criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 20, 0.05)
		self.lk_params = dict(winSize = (6, 6), maxLevel = 5, criteria = term_criteria)
		self.gf_params = dict(maxCorners = 100, qualityLevel = 0.3, minDistance = 7, blockSize = 7)

	def find_points(self, frame):
		corners = cv2.goodFeaturesToTrack(frame, **self.gf_params)
		prev_x_max = self.prev_bbox.x + self.prev_bbox.w
		prev_y_max = self.prev_bbox.y + self.prev_bbox.h

		points = []

		for corner in corners:
			x,y = corner.ravel()
			if (x < prev_x_max) and (x > self.prev_bbox.x) and (y < prev_y_max) and (y > self.prev_bbox.y):
				points.append((float(x), float(y)))

		if(len(points) == 0):
			self.find_points_backup()
		else:
			self.old_points = np.expand_dims(np.array(points).astype(np.float32),1)

	def find_points_backup(self):
		max_pts = 3

		step_x = max(int((self.prev_bbox.w)/max_pts), 1)
		step_y = max(int((self.prev_bbox.h)/max_pts), 1)
		points = []

		for y in range(self.prev_bbox.y, self.prev_bbox.y+self.prev_bbox.h, step_y):
			for x in range(self.prev_bbox.x, self.prev_bbox.x+self.prev_bbox.w, step_x):
				points.append((float(x), float(y)))

		self.old_points = np.expand_dims(np.array(points).astype(np.float32),1)

	def cross_correlation(self, last_frame, new_frame, status, err):
		for i in range(self.old_points.shape[0]):
			if status[i]==1:
				x1 = int(self.old_points[i][0][0])
				y1 = int(self.old_points[i][0][1])

				x2 = int(self.new_points[i][0][0])
				y2 = int(self.new_points[i][0][1])

				rec1 = cv2.getRectSubPix(last_frame, (10,10), (x1, y1))
				rec2 = cv2.getRectSubPix(new_frame, (10,10), (x2, y2))
				cross = cv2.matchTemplate(rec1, rec2, cv2.TM_CCOEFF_NORMED)

				cross_val = float(cross[0])
				err[i] = cross_val
			else:
				err[i] = 0.0
		return err

	def filter_points(self, status, err, reverse_err):
		err_med = np.median(err)
		one_left = False
		num = self.new_points.shape[0]

		new_opoints = []
		new_npoints = []
		new_rerr = []

		for i in range(num):
			if status[i] and (err[i] >= err_med):
				new_opoints.append(self.old_points[i])
				new_npoints.append(self.new_points[i])
				new_rerr.append(reverse_err[i])
				one_left = True

		if not one_left:
			return False

		self.old_points = np.array(new_opoints)
		self.new_points = np.array(new_npoints)
		reverse_err = np.array(new_rerr)

		reverse_med = np.median(reverse_err)
		one_left = False
		new_num = self.new_points.shape[0]

		new_opoints2 = []
		new_npoints2 = []

		for i in range(new_num):
			if status[i] and (reverse_err[i] <= reverse_med):
				new_opoints2.append(self.old_points[i])
				new_npoints2.append(self.new_points[i])
				one_left = True

		self.old_points = np.array(new_opoints2)
		self.new_points = np.array(new_npoints2)

		if one_left:
			return True
		else:
			return False

	def op_flow(self, last_frame, new_frame):
		self.new_points, status, err = cv2.calcOpticalFlowPyrLK(last_frame, new_frame,
																		 self.old_points, None,
																		 **self.lk_params)
		self.reverse_points, reverse_status, reverse_err = cv2.calcOpticalFlowPyrLK(new_frame, last_frame,
																		self.new_points, None,
																		**self.lk_params)
		for i in range(self.old_points.shape[0]):
			real = self.reverse_points[i][0][0] - self.old_points[i][0][0]
			imag = self.reverse_points[i][0][1] - self.old_points[i][0][1]
			reverse_err[i] = pow(real, 2) + pow(imag, 2)

		err = self.cross_correlation(last_frame, new_frame, status, err)
		return self.filter_points(status, err, reverse_err)

	def find_new_box(self):
		num = self.old_points.shape[0]

		x_diff = []
		y_diff = []

		for i in range(num):
			x_diff.append(self.new_points[i][0][0] - self.old_points[i][0][0])
			y_diff.append(self.new_points[i][0][1] - self.old_points[i][0][1])

		x_med = np.median(np.array(x_diff))
		y_med = np.median(np.array(y_diff))

		scaling_factor = 1.0
		if (num > 1):
			dist = []
			for i in range(num):
				for j in range(i+1, num):
					top = np.sqrt(pow(self.new_points[i][0][0] - self.new_points[j][0][0], 2) + pow(self.new_points[i][0][1] - self.new_points[j][0][1], 2))
					bot = np.sqrt(pow(self.old_points[i][0][0] - self.old_points[j][0][0], 2) + pow(self.old_points[i][0][1] - self.old_points[j][0][1], 2))

					dist.append(top/bot)

			scaling_factor = np.median(dist)

		d1 = 0.5 * (scaling_factor - 1) * self.prev_bbox.w
		d2 = 0.5 * (scaling_factor - 1) * self.prev_bbox.h

		new_x = max(int(np.round(self.prev_bbox.x + x_med - d1)), 0)
		new_y = max(int(np.round(self.prev_bbox.y + y_med - d2)), 0)
		new_w = int(np.round(self.prev_bbox.w * scaling_factor))
		new_h = int(np.round(self.prev_bbox.h * scaling_factor))

		self.new_bbox.set_values(new_x, new_y, new_w, new_h)

	def process_frame(self, last_frame, new_frame):
		self.old_points = None
		self.new_points = None

		next_bbox = bbox()

		self.find_points(last_frame)

		if len(self.old_points) < 5:
			self.tracking = False
		else:
			self.tracking = self.op_flow(last_frame, new_frame)

			if self.tracking:
				self.find_new_box()


		if (self.tracking):
			next_bbox = self.new_bbox
			self.prev_bbox = next_bbox

		return next_bbox, self.tracking

class Test_optical_flow(unittest.TestCase):
	def test_find_points(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		opflow = optical_flow(test_box)

		opflow.find_points(img)

		self.assertEqual(len(opflow.old_points), 16)

	def test_filter_points(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		opflow = optical_flow(test_box)

		status = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		err = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		reverse_err = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

		opflow.find_points(img)

		opflow.new_points = opflow.old_points

		bool = opflow.filter_points(status, err, reverse_err)
		self.assertEqual(len(opflow.old_points), 16)
		self.assertEqual(bool, True)

	def test_filter_points_None(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)

		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))
		img = np.asarray(img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		opflow = optical_flow(test_box)

		status = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		err = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		reverse_err = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

		opflow.find_points(img)

		opflow.new_points = opflow.old_points

		bool = opflow.filter_points(status, err, reverse_err)
		self.assertEqual(bool, False)

	def test_op_flow(self):
		test_box = bbox()
		test_box.set_values(0, 0, 100, 100)

		opflow = optical_flow(test_box)
		directory = os.path.dirname(__file__)

		img = Image.open(os.path.join(directory, 'test_images/sample_image_1.png'))

		img = np.asarray(img)
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		opflow.find_points(img)

		output = opflow.op_flow(img, img)
		self.assertEqual(output, False)


if __name__=='__main__':
	unittest.main()



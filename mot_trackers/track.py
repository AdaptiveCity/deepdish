import numpy as np

from .kalman_filter import KalmanFilter
import unittest

def to_xyah(box):
	ret = np.array(box.copy(), dtype=float)
	ret[0] += ret[2]/2
	ret[1] += ret[3]/2

	ret[2] = ret[2] * 1.0
	ret[2] /= ret[3]

	return ret

def to_xywh(box):
	ret = box[:4].copy()
	ret[2] *= ret[3]

	ret[0] -= ret[2]/2
	ret[1] -= ret[3]/2

	return ret

class sort_track:
	def __init__(self, track_id, frame_id, box, confidence):
		self.id = track_id
		self.loss = 0

		self.kf = KalmanFilter()
		mean, covariance = self.kf.initiate(to_xyah(box))

		self.mean =  mean
		self.covariance = covariance

		self.update(frame_id, box, confidence, False)

	def update(self, frame_id, box, confidence, lost):
		self.box = np.array(box)
		self.confidence = confidence

		self.frame_id = frame_id

		if (lost):
			self.lost += 1
		else:
			self.lost = 0

		self.mean, self.covariance = self.kf.update(self.mean, self.covariance, to_xyah(self.box))

	def get_output(self):
		bbox = self.box
		output = (self.frame_id, self.id, bbox[0], bbox[1], bbox[2], bbox[3], self.confidence, -1, -1, -1)
		return output

	def predict(self):
		self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
		vector = to_xywh(self.mean)

		return np.array([vector[0], vector[1], vector[2], vector[3]])

class centroid_track:
	def __init__(self, track_id, frame_id, box, confidence):
		self.id = track_id
		self.loss = 0

		newbox = np.array([box[0] + 0.5*box[2], box[1] + 0.5*box[3]])

		self.kf = KalmanFilter()
		mean, covariance = self.kf.initiate(to_xyah(box))

		self.mean =  mean
		self.covariance = covariance

		self.update(frame_id, box, confidence, False)

	def update(self, frame_id, box, confidence, lost):
		self.box = np.array(box)
		self.confidence = confidence

		self.frame_id = frame_id

		if (lost):
			self.lost += 1
		else:
			self.lost = 0

		cent = np.array((self.box[0] + 0.5*self.box[2], self.box[1] + 0.5*self.box[3]))
		#self.kf.update(cent)
		self.mean, self.covariance = self.kf.update(self.mean, self.covariance, to_xyah(box))

	def get_output(self):
		bbox = self.box
		output = (self.frame_id, self.id, bbox[0], bbox[1], bbox[2], bbox[3], self.confidence, -1, -1, -1)
		return output

	def predict(self):
		self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
		vector = to_xywh(self.mean)

		return np.array([vector[0], vector[1], vector[2], vector[3]])

class Test_track(unittest.TestCase):
	def test_to_xyah(self):
		test_box = [0, 10, 100, 200]

		new_box = to_xyah(test_box)
		self.assertEqual(new_box[0], 50.0)
		self.assertEqual(new_box[1], 110.0)
		self.assertEqual(new_box[2], 0.5)
		self.assertEqual(new_box[3], 200.0)

	def test_to_xywh(self):
		test_box = [50.0, 110.0, 0.5, 200.0]

		new_box = to_xywh(test_box)
		self.assertEqual(new_box[0], 0.0)
		self.assertEqual(new_box[1], 10.0)
		self.assertEqual(new_box[2], 100.0)
		self.assertEqual(new_box[3], 200.0)

	def test_update_sort(self):
		test_box = [0, 10, 100, 200]
		test_sort = sort_track(1, 1, test_box, 0.5)

		new_box = [0, 10, 150, 200]
		test_sort.update(2, new_box, 0.8, True)
		self.assertEqual(test_sort.frame_id, 2)
		self.assertEqual(test_sort.box[2], 150)
		self.assertEqual(test_sort.lost, 1)
		self.assertEqual(test_sort.confidence, 0.8)

	def test_update_centroid(self):
		test_box = [0, 10, 100, 200]
		test_centroid = centroid_track(1, 1, test_box, 0.5)

		new_box = [0, 10, 150, 200]
		test_centroid.update(2, new_box, 0.8, True)
		self.assertEqual(test_centroid.frame_id, 2)
		self.assertEqual(test_centroid.box[2], 150)
		self.assertEqual(test_centroid.lost, 1)
		self.assertEqual(test_centroid.confidence, 0.8)


if __name__=='__main__':
	unittest.main()
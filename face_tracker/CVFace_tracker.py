# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import cv2
from PIL import Image

import unittest

class CVFace_tracker:

	def __init__(self, model, max_age):
		self.model = model
		self.max_age = max_age
		self.tracker_types = ['BOOSTING', 'MIL', 'TLD', 'KCF']
		#TLD too unstable, BOOSTING and MIL slow, use KCF

		self.tracks = []
		self.deleted_tracks = []
		self._next_id = 1

	def create_tracker(self):
		if self.model == self.tracker_types[0]:
			tracker = cv2.TrackerBoosting_create()
		elif self.model == self.tracker_types[1]:
			tracker = cv2.TrackerMIL_create()
		elif self.model == self.tracker_types[2]:
			tracker = cv2.TrackerTLD_create()
		elif self.model == self.tracker_types[3]:
			tracker = cv2.TrackerKCF_create()
		else:
			tracker = None
			print('Incorrect tracker name')
			print('Available trackers are:')
			for t in self.tracker_types:
				print(t)
		return tracker

	def add_tracker(self, frame, face, max_x, max_y):
		(x,y,w,h) = list(map(int, face[:4]))
		x, y = int(np.clip(x,0,max_x)), int(np.clip(y,0,max_y))
		w, h = int(np.clip(w,0,max_x-x)), int(np.clip(h,0,max_y-y))
		tracker = self.create_tracker()
		tracker.init(frame,(x,y,w,h))
		self.tracks.append(tracker)

	def update_tracks(self, frame):
		faces = []
		allFound = True
		for track in self.tracks:
			ok,bbox=track.update(frame)
			if ok:
				(x,y,w,h)=[int(v) for v in bbox]
				faces.append([x,y,w,h])
			else:
				print("REMOVED")
				self.tracks.remove(track)
				self.deleted_tracks.append(track)
				allFound = False
				break

		return (faces, allFound)

	def clear_tracks(self):
		self.tracks = []

class Test_CVFace_tracker(unittest.TestCase):
	def test_create_tracker_KCF(self):
		test_tracker = CVFace_tracker('KCF', 10)
		img = Image.open('modelfiles\sample_image_1.png')
		width, height = img.size

		face = [10, 20, 30, 15]

		test_tracker.add_tracker(img, face, width, height)
		track = test_tracker.tracks[0]

		self.assertEqual(type(track), cv2.TrackerKCF)

	def test_create_tracker_BOOSTING(self):
		test_tracker = CVFace_tracker('BOOSTING', 10)
		img = Image.open('modelfiles\sample_image_1.png')
		width, height = img.size

		face = [10, 20, 30, 15]

		test_tracker.add_tracker(img, face, width, height)
		track = test_tracker.tracks[0]

		self.assertEqual(type(track), cv2.TrackerBoosting)

	def test_create_tracker_MIL(self):
		test_tracker = CVFace_tracker('MIL', 10)
		img = Image.open('modelfiles\sample_image_1.png')
		width, height = img.size

		face = [10, 20, 30, 15]

		test_tracker.add_tracker(img, face, width, height)
		track = test_tracker.tracks[0]

		self.assertEqual(type(track), cv2.TrackerMIL)

	def test_create_tracker_TLD(self):
		test_tracker = CVFace_tracker('TLD', 10)
		img = Image.open('modelfiles\sample_image_1.png')
		width, height = img.size

		face = [10, 20, 30, 15]

		test_tracker.add_tracker(img, face, width, height)
		track = test_tracker.tracks[0]

		self.assertEqual(type(track), cv2.TrackerTLD)

	def test_clear_tracks(self):
		test_tracker = CVFace_tracker('KCF', 10)
		img = Image.open('modelfiles\sample_image_1.png')
		width, height = img.size

		face = [10, 20, 30, 15]

		test_tracker.add_tracker(img, face, width, height)

		test_tracker.clear_tracks()

		self.assertEqual(test_tracker.tracks, [])

	def test_update_failed(self):
		test_tracker = CVFace_tracker('KCF', 10)
		img = Image.open('modelfiles\sample_image_1.png')
		img2 = Image.open('modelfiles\sample_image_2.png')
		width, height = img.size

		face = [10, 20, 30, 15]

		test_tracker.add_tracker(img, face, width, height)

		(faces, allFound) = test_tracker.update_tracks(img2)

		self.assertEqual(allFound, False)

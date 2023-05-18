# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np

from .camshift import Camshift
from .bbox import bbox

class CAMSHIFT_tracker:

	def __init__(self, model, max_age):
		self.model = model
		self.max_age = max_age

		self.tracks = []
		self.deleted_tracks = []
		self._next_id = 1

	def create_tracker(self, face, frame):
		tracker = Camshift(face, frame)
		return tracker

	def add_tracker(self, frame, face, max_x, max_y):
		(x,y,w,h) = list(map(int, face[:4]))
		x, y = int(np.clip(x,0,max_x)), int(np.clip(y,0,max_y))
		w, h = int(np.clip(w,0,max_x-x)), int(np.clip(h,0,max_y-y))

		pbox = bbox()
		pbox.set_values(x, y, w, h)

		track = self.create_tracker(pbox, frame)

		tracker = ((x,y,w,h), track, 0)

		self.tracks.append(tracker)

	def update_tracks(self, frame):
		faces = []
		allFound = True

		for i in range(len(self.tracks)):
			(track_window, tracker, age) = self.tracks[i]

			tracker.compute_bbox(frame)
			pts = tracker.get_rotated()
			#print(pts)
			#pts = cv2.boxPoints(ret2)
			#pts = np.int0(pts)
			xpts = [p[0] for p in pts]
			ypts = [p[1] for p in pts]
			xmax = max(xpts)
			xmin = min(xpts)
			ymax = max(ypts)
			ymin = min(ypts)

			x = xmin
			y = ymin
			w = int(xmax - xmin)
			h = int(ymax - ymin)

			(x, y, w, h) = tracker.get_bbox()

			self.tracks[i] = ((x, y, w, h), tracker, (age+1))
			if((age + 1) > self.max_age):
				allFound = False
			faces.append([x,y,w,h])

		return (faces, allFound)

	def clear_tracks(self):
		self.tracks = []
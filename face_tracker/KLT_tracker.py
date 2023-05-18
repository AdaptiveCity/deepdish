# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import cv2
import time

from .optical_flow import optical_flow
from .bbox import bbox

class KLT_tracker:

    def __init__(self, model, max_age):
        self.model = model
        self.max_age = max_age
        self.tracks = []
        self.deleted_tracks = []
        self._next_id = 1

    def create_tracker(self, old_bbox):
        tracker = optical_flow(old_bbox)
        return tracker

    def add_tracker(self, frame, face, max_x, max_y):
    	(x,y,w,h) = list(map(int, face[:4]))
    	x, y = int(np.clip(x,0,max_x)), int(np.clip(y,0,max_y))
    	w, h = int(np.clip(w,0,max_x-x)), int(np.clip(h,0,max_y-y))

    	frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    	pbox = bbox()
    	pbox.set_values(x, y, w, h)
    	track = self.create_tracker(pbox)
    	#track.get_last_box(pbox)
    	tracker = (track, frameGray, pbox, 0)

    	self.tracks.append(tracker)

    def update_tracks(self, frame):
    	faces = []
    	allFound = True
    	new_tracks = []
    	frameGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    	for i in range(len(self.tracks)):
    		(tracker, old_frame, pbox, age) = self.tracks[i]

    		new_pbox, not_lost = tracker.process_frame(old_frame, frameGray)

    		faces.append(new_pbox.get_values())

    		if((age + 1) > self.max_age):
    		    allFound = False

    		self.tracks[i] = (tracker, frameGray, new_pbox, (age+1))

    		allFound = allFound & not_lost


    	return (faces, allFound)

    def clear_tracks(self):
    	self.tracks = []
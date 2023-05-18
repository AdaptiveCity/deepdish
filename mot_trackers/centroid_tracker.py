import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from .track import centroid_track
import unittest

def calculate_centroid(boxlist):
	xlist = boxlist[:, 0]
	ylist = boxlist[:, 1]
	wlist = boxlist[:, 2]
	hlist = boxlist[:, 3]

	new_x = xlist + 0.5*wlist
	new_y = ylist + 0.5*hlist

	return np.hstack([new_x[:, None], new_y[:, None]])

def assign_tracks(tracks, detections, threshold = 30):

	if (tracks.size == 0) or (detections.size == 0):
			new_detection_ids = np.arange(len(detections), dtype=int)
			return np.zeros((0,2)), new_detection_ids, np.zeros((0, len(tracks)))

	t_unmatched, d_unmatched = [], []
	final_matches = []

	t_centroids = calculate_centroid(tracks)
	d_centroids = calculate_centroid(detections)
	cost_matrix = cdist(t_centroids, d_centroids)
	#print(cost_matrix)

	t_matches, d_matches = linear_sum_assignment(cost_matrix)

	for track, detection in zip(t_matches, d_matches):
		if cost_matrix[track, detection] > threshold:
			t_unmatched.append(track)
			d_unmatched.append(detection)
		else:
			final_matches.append((track, detection))

	for track in range(tracks.shape[0]):
		if track not in t_matches:
			t_unmatched.append(track)

	for detection in range(detections.shape[0]):
		if detection not in d_matches:
			d_unmatched.append(detection)

	if (len(final_matches) != 0):
		final_matches = np.array(final_matches)
	else:
		final_matches = np.zeros((0,2), dtype=int)

	#print(final_matches)

	return final_matches, np.array(d_unmatched), np.array(t_unmatched)

class centroidtracker:
	def __init__(self, lost_threshold = 5, centroid_threshold = 30):
		self.centroid_threshold = centroid_threshold
		self.lost_threshold = lost_threshold

		self.tracks = dict()

		self.next_id = 0
		self.frame_count = 0

	def add_track(self, frame_id, box, confidence):
		self.tracks[self.next_id] = centroid_track(self.next_id, frame_id, box, confidence)
		self.next_id += 1

	def update(self, boxes, scores):
		self.frame_count +=1
		detections = np.array(boxes, dtype=int)

		track_ids = list(self.tracks.keys())
		tracks = []

		for id in track_ids:
			predicted_box = self.tracks[id].predict()
			tracks.append(predicted_box)

		tracks = np.array(tracks)

		if len(detections) == 0:
			for i in range(len(track_ids)):
				id = track_ids[i]
				box = tracks[i, :]

				confidence = self.tracks[id].confidence
				self.tracks[id].update(self.frame_count, box, confidence, True)
				if self.tracks[id].lost > self.lost_threshold:
					del self.tracks[id]

		else:
			matches, d_unmatched, t_unmatched = assign_tracks(tracks, detections, self.centroid_threshold)
			#print(matches, d_unmatched, t_unmatched)

			for i in range(matches.shape[0]):
				track, detection = matches[i, :]
				id = track_ids[track]
				box = boxes[detection, :]
				confidence = scores[detection]
				self.tracks[id].update(self.frame_count, box, confidence, False)

			for detection in d_unmatched:
				box = boxes[detection, :]
				confidence = scores[detection]
				self.add_track(self.frame_count, box, confidence)

			for track in t_unmatched:
				id = track_ids[track]
				box = tracks[track, :]
				confidence = self.tracks[id].confidence
				self.tracks[id].update(self.frame_count, box, confidence, True)
				if self.tracks[id].lost > self.lost_threshold:
					del self.tracks[id]

		outputs = []
		for id, track in self.tracks.items():
			if not track.lost:
				outputs.append(track.get_output())
		return outputs

class Test_centroid_track(unittest.TestCase):
	def test_calculate_centroid(self):
		box1 = [0, 10, 100, 200]
		box_list = [box1]
		box_list = np.array(box_list)

		centroid = calculate_centroid(box_list)

		self.assertEqual(centroid[0][0], 50)

	def test_assign_tracks(self):
		box1 = [0, 10, 100, 200]
		box2 = [0, 10, 80, 250]

		tracks = []
		tracks.append(box1)
		tracks = np.array(tracks)

		detections = []
		detections.append(box2)
		detections = np.array(detections)

		matches, d_unmatched, t_unmatched = assign_tracks(tracks, detections, 30)

		self.assertEqual(len(matches), 1)

	def test_assign_tracks_fail(self):
		box1 = [0, 10, 100, 200]
		box2 = [200, 400, 80, 250]

		tracks = []
		tracks.append(box1)
		tracks = np.array(tracks)

		detections = []
		detections.append(box2)
		detections = np.array(detections)

		matches, d_unmatched, t_unmatched = assign_tracks(tracks, detections, 30)

		self.assertEqual(len(matches), 0)
		self.assertEqual(len(d_unmatched), 1)

if __name__=='__main__':
	unittest.main()
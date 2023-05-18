import numpy as np
from scipy.optimize import linear_sum_assignment
from .track import sort_track
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

	final_x1 = min(x11, x12)
	final_y1 = min(y11, y12)

	if(final_x1 - final_x0 <= 0) or (final_y1 - final_y0 <= 0):
		return 0.0

	intersection = (final_x1 - final_x0) * (final_y1 - final_y0)

	union1 = (x11 - x01) * (y11 - y01)
	union2 = (x12 - x02) * (y12 - y02)

	union = union1 + union2 - intersection

	return intersection / union

def assign_tracks(tracks, detections, threshold = 0.5):

	if (tracks.size == 0) or (detections.size == 0):
		new_detection_ids = np.arange(len(detections), dtype=int)
		return np.zeros((0,2)), new_detection_ids, np.zeros((0, len(tracks)))

	t_unmatched, d_unmatched = [], []
	final_matches = []

	cost_matrix = np.zeros((tracks.shape[0], detections.shape[0]))

	for track in range(tracks.shape[0]):
		for detection in range(detections.shape[0]):
			cost_matrix[track, detection] = calculate_iou(tracks[track, :], detections[detection, :])

	#print(cost_matrix)

	t_matches, d_matches = linear_sum_assignment(-cost_matrix)

	for track, detection in zip(t_matches, d_matches):
		if cost_matrix[track, detection] < threshold:
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

	return final_matches, np.array(d_unmatched), np.array(t_unmatched)

class sorttracker:
	def __init__(self, lost_threshold = 5, iou_threshold = 0.4):
		self.iou_threshold = iou_threshold
		self.lost_threshold = lost_threshold

		self.tracks = dict()

		self.next_id = 0
		self.frame_count = 0

	def add_track(self, frame_id, box, confidence):
		self.tracks[self.next_id] = sort_track(self.next_id, frame_id, box, confidence)
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
			matches, d_unmatched, t_unmatched = assign_tracks(tracks, detections, self.iou_threshold)
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
					#print("DELETING " + str(id))

		outputs = []
		for id, track in self.tracks.items():
			if not track.lost:
				outputs.append(track.get_output())
		return outputs

class Test_sort_track(unittest.TestCase):
	def test_calculate_iou(self):
		box1 = [0, 10, 100, 200]
		box2 = [20, 40, 80, 250]

		iou = calculate_iou(box1, box2)
		over_limit = (iou > 0.5)

		self.assertEqual(over_limit, True)

	def test_calculate_iou_fail(self):
		box1 = [0, 10, 100, 200]
		box2 = [200, 400, 80, 250]

		iou = calculate_iou(box1, box2)
		over_limit = (iou > 0.5)

		self.assertEqual(over_limit, False)

	def test_assign_tracks(self):
		box1 = [0, 10, 100, 200]
		box2 = [20, 40, 80, 250]

		tracks = []
		tracks.append(box1)
		tracks = np.array(tracks)

		detections = []
		detections.append(box2)
		detections = np.array(detections)

		matches, d_unmatched, t_unmatched = assign_tracks(tracks, detections, 0.5)

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

		matches, d_unmatched, t_unmatched = assign_tracks(tracks, detections, 0.5)

		self.assertEqual(len(matches), 0)
		self.assertEqual(len(d_unmatched), 1)

if __name__=='__main__':
	unittest.main()
"""Adaptor from tflite_objector_detector to simplified interface used by deepdish"""
import sys
import time
import numpy as np

from tools.tflite_object_detector import ObjectDetector
from tools.tflite_object_detector import ObjectDetectorOptions

class TFLITE:
    def __init__(self, wanted_labels=None, model_file=None, label_file=None, num_threads=None, edgetpu=False, libedgetpu=None, score_threshold=0.5):
        self.opts = ObjectDetectorOptions(
            num_threads=num_threads or 1,
            score_threshold=score_threshold,
            enable_edgetpu=edgetpu)
        self.use_edgetpu = edgetpu
        self.num_threads = num_threads or 1
        self.detector = ObjectDetector(model_path=model_file, options=self.opts)
        if wanted_labels is None:
            wanted_labels = ['person']
        self.wanted_labels = wanted_labels
        self.label_list = self.detector._label_list
        # self.labels is used externally by deepdish
        self.labels = {i+1: self.label_list[i] for i in range(0, len(self.label_list))}
        self.width, self.height = self.detector.input_size

    def detect_image(self, img):
        dets = self.detector.detect(np.array(img)[...,:3])
        return_boxs = []
        return_lbls = []
        return_scrs = []
        for det in dets:
            lblscrs = [(w, c.score)
                       for c in det.categories
                       for w in self.wanted_labels
                       if c.label == w]
            if lblscrs:
                b = det.bounding_box
                return_boxs.append([b.left, b.top, b.right - b.left, b.bottom - b.top])
                return_lbls.append(lblscrs[0][0])
                return_scrs.append(lblscrs[0][1])
        return (return_boxs, return_lbls, return_scrs)

# -*- coding: utf-8 -*-
# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3

import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from PIL import Image, ImageFont, ImageDraw

class BoundBox:
        def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
                self.xmin = xmin
                self.ymin = ymin
                self.xmax = xmax
                self.ymax = ymax
                self.objness = objness
                self.classes = classes
                self.label = -1
                self.score = -1

        def get_label(self):
                if self.label == -1:
                        self.label = np.argmax(self.classes)

                return self.label

        def get_score(self):
                if self.score == -1:
                        self.score = self.classes[self.get_label()]

                return self.score

        def get_tlbr(self):
                return np.array([self.xmin,self.ymin,self.xmax,self.ymax])

def _sigmoid(x):
        return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5
        boxes = []
        netout[..., :2]  = _sigmoid(netout[..., :2])
        netout[..., 4:]  = _sigmoid(netout[..., 4:])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
                row = i / grid_w
                col = i % grid_w
                for b in range(nb_box):
                        # 4th element is objectness score
                        objectness = netout[int(row)][int(col)][b][4]
                        if(objectness.all() <= obj_thresh): continue
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[int(row)][int(col)][b][:4]
                        x = (col + x) / grid_w # center position, unit: image width
                        y = (row + y) / grid_h # center position, unit: image height
                        w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                        h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
                        # last elements are class probabilities
                        classes = netout[int(row)][col][b][5:]
                        box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                        boxes.append(box)
        return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
        new_w, new_h = net_w, net_h
        for i in range(len(boxes)):
                x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
                y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
                boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
                boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
                boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
                boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
                if x4 < x1:
                        return 0
                else:
                        return min(x2,x4) - x1
        else:
                if x2 < x3:
                         return 0
                else:
                        return min(x2,x4) - x3

def bbox_iou(box1, box2):
        intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        intersect = intersect_w * intersect_h
        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        union = w1*h1 + w2*h2 - intersect
        return float(intersect) / union

def do_nms(boxes, nms_thresh):
        if len(boxes) > 0:
                nb_class = len(boxes[0].classes)
        else:
                return
        for c in range(nb_class):
                sorted_indices = np.argsort([-box.classes[c] for box in boxes])
                for i in range(len(sorted_indices)):
                        index_i = sorted_indices[i]
                        if boxes[index_i].classes[c] == 0: continue
                        for j in range(i+1, len(sorted_indices)):
                                index_j = sorted_indices[j]
                                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                                        boxes[index_j].classes[c] = 0

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
        v_boxes, v_labels, v_scores = list(), list(), list()
        # enumerate all boxes
        for box in boxes:
                # enumerate all possible labels
                for i in range(len(labels)):
                        # check if the threshold for this label is high enough
                        if box.classes[i] > thresh:
                                v_boxes.append(box)
                                v_labels.append(labels[i])
                                v_scores.append(box.classes[i]*100)
                                # don't break, many labels may trigger for one box
        return v_boxes, v_labels, v_scores

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128,128,128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image

class YOLO(object):
    def __init__(self,wanted_labels=None,model_file=None,label_file=None,num_threads=None,score_threshold=0.5):
        basedir = os.getenv('DEEPDISHHOME','.')
        self.model_path = '{}/detectors/yolo/yolo.h5'.format(basedir)
        self.model = load_model(self.model_path)
        self.anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
        self.class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
                            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
                            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                            "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                            "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
                            "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                            "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
                            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        self.score_threshold = score_threshold
        self.iou = 0.5
        #self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        #self.boxes, self.scores, self.classes = self.generate()
        if wanted_labels is None:
            wanted_labels = ['person']
        self.wanted_labels = wanted_labels
        self.labels = dict(enumerate(self.class_names))

    def detect_image(self, image):
        image_w, image_h = image.size
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        yhat = self.model.predict(image_data)
        input_w, input_h = self.model_image_size
        boxes = list()
        for i in range(len(yhat)):
            # decode the output of the network
            boxes += decode_netout(yhat[i][0], self.anchors[i], self.score_threshold, input_h, input_w)
        # correct the sizes of the bounding boxes for the shape of the image
        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # suppress non-maximal boxes
        do_nms(boxes, 0.5)

        v_boxes, v_classes, v_scores = get_boxes(boxes, self.class_names, self.score_threshold)

        return_boxs = []
        return_lbls = []
        return_scrs = []
        for i, bbox in reversed(list(enumerate(v_boxes))):
            predicted_class = self.class_names[bbox.get_label()]
            box = bbox.get_tlbr()
            score = bbox.get_score()
            print('{} ({}) at {}'.format(predicted_class, score, box))
            if predicted_class not in self.wanted_labels:
                continue
            if score < self.score_threshold:
                continue
            x = int(box[1])
            y = int(box[0])
            w = int(box[3]-box[1])
            h = int(box[2]-box[0])
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0
            return_boxs.append([x,y,w,h])
            return_lbls.append(predicted_class)
            return_scrs.append(score)

        return (return_boxs, return_lbls, return_scrs)

    def close_session(self):
        pass

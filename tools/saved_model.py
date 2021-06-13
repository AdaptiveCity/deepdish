import time
import os
import tensorflow as tf
import numpy as np
import cv2
from google.protobuf import text_format
import tools.string_int_label_map_pb2 as string_int_label_map_pb2

class SAVED_MODEL:
    def __init__(self, wanted_labels=None, model_file=None, label_file=None, num_threads=None, edgetpu=False, libedgetpu=None, score_threshold=0.5):
        basedir = os.getenv('DEEPDISHHOME','.')
        if model_file is None:
            model_file = os.path.join(basedir, 'detectors/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/saved_model')
        if label_file is None:
            label_file = os.path.join(basedir, 'detectors/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/label_map.pbtxt')
        if wanted_labels is None:
            wanted_labels = ['person']
        self.wanted_labels = wanted_labels
        self.label_file = label_file
        self.score_threshold = score_threshold
        self.labels = self._get_labels()
        self.detect_fn = tf.saved_model.load(model_file)

    def _get_labels(self):
        # object_detection package deps are broken - use imported code for time being
        #tab = label_map_util.create_category_index_from_labelmap(self.label_file, use_display_name=True)
        #return {i: cat['name'] for i, cat in tab.items()}
        # imported code:
        label_map = load_labelmap(self.label_file)
        return {e.id: e.display_name for e in label_map.ListFields()[0][1]}

    def detect_image(self, img):
        w, h = img.size
        image_np = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        return_boxs = [] # List of (x, y, w, h) in image coordinates
        return_lbls = [] # text labels (display names)
        return_scrs = [] # scores from 0 to 1
        for i in range(num_detections):
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i] # normalised range [0,1]
            xywh = [int(xmin * w), int(ymin * h), int((xmax - xmin) * w), int((ymax - ymin) * h)]
            cls = detections['detection_classes'][i]
            scr = detections['detection_scores'][i]
            if scr < self.score_threshold:
                break
            lbl = self.labels[cls]
            if lbl in self.wanted_labels:
                return_boxs.append(xywh)
                return_lbls.append(lbl)
                return_scrs.append(scr)

        return (return_boxs, return_lbls, return_scrs)


def load_labelmap(path):
  """Loads label map proto.

  Args:
    path: path to StringIntLabelMap proto text file.
  Returns:
    a StringIntLabelMapProto
  """
  with tf.io.gfile.GFile(path, 'r') as fid:
    label_map_string = fid.read()
    label_map = string_int_label_map_pb2.StringIntLabelMap()
    try:
      text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
      label_map.ParseFromString(label_map_string)
  _validate_label_map(label_map)
  return label_map

def _validate_label_map(label_map):
  """Checks if a label map is valid.

  Args:
    label_map: StringIntLabelMap to validate.

  Raises:
    ValueError: if label map is invalid.
  """
  for item in label_map.item:
    if item.id < 0:
      raise ValueError('Label map ids should be >= 0.')
    if (item.id == 0 and item.name != 'background' and
        item.display_name != 'background'):
      raise ValueError('Label map id 0 is reserved for the background label')

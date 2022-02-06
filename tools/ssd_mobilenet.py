import time
import os
import platform
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor
import numpy as np
# pylint: disable=g-import-not-at-top
try:
  # Import TFLite interpreter from tflite_runtime package if it's available.
  from tflite_runtime.interpreter import Interpreter
  from tflite_runtime.interpreter import load_delegate
except ImportError:
  # If not, fallback to use the TFLite interpreter from the full TF package.
  import tensorflow as tf

  Interpreter = tf.lite.Interpreter
  load_delegate = tf.lite.experimental.load_delegate
# pylint: enable=g-import-not-at-top

def edgetpu_lib_name():
  """Returns the library name of EdgeTPU in the current platform."""
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)

class SSDMobileNet:
  def __init__(self,model_path,label_path,num_threads=None,edgetpu=False,libedgetpu=None,score_threshold=0.5):
    if edgetpu:
      if libedgetpu is None: libedgetpu = edgetpu_lib_name()
      delegate = load_delegate(libedgetpu)
      self.interpreter = Interpreter(model_path, num_threads=num_threads, experimental_delegates=[delegate])
    else:
      self.interpreter = Interpreter(model_path, num_threads=num_threads)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.height = self.input_details[0]['shape'][1]
    self.width = self.input_details[0]['shape'][2]
    self.labels = self.load_labels(label_path)

  def load_labels(self,path):
    with open(path, 'r') as f:
      return {i: line.strip() for i, line in enumerate(f.readlines())}

  # returns (input_image, original_image)
  def prepare_image_path(self, image_path):
    img = Image.open(image_path)
    return self.prepare_image(img), img

  def prepare_image(self, img):
    img_resized = img.convert('RGB').resize((self.width, self.height), Image.ANTIALIAS)
    #img_resized = img_resized.astype(np.float32)
    return img_resized

  def nms_boxes(self, boxes, labels, scores, iou_threshold):
    nboxes, nlabels, nscores = [], [], []
    for c in set(labels):
      inds = np.where(labels == c)
      b = boxes[inds]
      c = labels[inds]
      s = scores[inds]

      x = b[:, 0]
      y = b[:, 1]
      w = b[:, 2] - b[:, 0]
      h = b[:, 3] - b[:, 1]

      areas = w * h
      order = s.argsort()[::-1]

      keep = []
      while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 1)
        h1 = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

      keep = np.array(keep)

      nboxes.append(b[keep])
      nlabels.append(c[keep])
      nscores.append(s[keep])
    return nboxes, nlabels, nscores

  def predict(self, image, confidence=0.5, iou_threshold=0.5, original_image_size=None):
    """Returns a sorted array of classification results."""
    self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(image, 0).astype(np.uint8))
    self.interpreter.invoke()
    output_arrays = self.interpreter.get_output_details()
    output_details = output_arrays[0]
    #output = np.squeeze(interpreter.get_tensor(output_details['index']))
    output = []
    for i in range(4):
      output.append(np.squeeze(self.interpreter.get_tensor(output_arrays[i]['index'])))

    # Check boxes for NaNs
    indices = np.where(np.isnan(output[0]))
    output[2][np.reshape(indices,-1)] = 0 # Zero-out the NaNs for the purpose of this test
    # Check scores for NaNs
    indices = np.where(np.isnan(output[2]))
    output[2][indices] = 0 # Zero-out the NaNs for the purpose of this test

    # Obtain indices of sufficiently certain object identifications
    indices = np.where(output[2] >= confidence)

    reorder = [1,0,3,2]
    if original_image_size is not None:
      w, h = original_image_size
    else:
      w, h = (self.width, self.height)

    boxes = output[0][indices][:,reorder] * [w,h,w,h]
    labels = output[1][indices]
    scores = output[2][indices]
    # If the model is quantized (uint8 data), then dequantize the results
    #if output_details['dtype'] == np.uint8:
      #scale, zero_point = output_details['quantization']
      #output = scale * (output - zero_point)

    #ordered = np.argpartition(-output, top_k)
    #return [(i, output[i]) for i in ordered[:top_k]]
    n_boxes, n_labels, n_scores = self.nms_boxes(boxes, labels, scores, iou_threshold)
    if n_boxes:
      boxes = np.concatenate(n_boxes)
      labels = np.concatenate(n_labels).astype(np.uint)
      scores = np.concatenate(n_scores)
      labelnames = []
      for lblidx in labels:
          if lblidx >= 0 and lblidx < len(self.labels) - 1:
              labelnames.append(self.labels[lblidx+1])
          else:
              print("Invalid label index: {} in {}".format(lblidx, labels))
      return boxes, labelnames, scores
    else:
      return [], [], []

  def draw_boxes_and_save(self, boxes, labels, scores, img, output_path):
    self.draw_boxes(boxes, labels, scores, img)
    img.save(output_path)

  def draw_boxes(self, boxes, labels, scores, img):
    draw = ImageDraw.Draw(img)
    original_size = np.array(img.size)
    detection_size = np.array((self.width, self.height))
    color = tuple(np.random.randint(0, 256, 3))
    for box, score, lbl in zip(boxes, scores, labels):
      #ratio = original_size / detection_size
      #rbox = list((box.reshape(2,2) * ratio).reshape(-1))
      #print(rbox)
      rbox=list(box)
      draw.rectangle(rbox, outline=color)
      draw.text(rbox[:2], '{} {:.2f}%'.format(lbl, score * 100), fill=color)


def test():
  t0 = time.time()
  ssdm = SSDMobileNet(os.path.join(os.getcwd(), 'coco/detect.tflite'),
                      os.path.join(os.getcwd(), 'coco/labelmap.txt'))
  t1 = time.time()
  inp, img = ssdm.prepare_image_path('example.jpg')
  boxes, labels, scores = ssdm.predict(inp,original_image_size=img.size)
  t2 = time.time()
  print(boxes, labels, scores)
  print("load time={}, inference time={}".format(t1 - t0, t2 - t1))
  ssdm.draw_boxes_and_save(boxes, labels, scores, img, 'output.jpg')

class SSD_MOBILENET():
  def __init__(self, wanted_labels=None, model_file=None, label_file=None, num_threads=None, edgetpu=False, libedgetpu=None, score_threshold=0.5):
    if model_file is None:
      model_file = 'ssd_mobilenet.tflite'
    if label_file is None:
      label_file = 'coco_labelmap.txt'
    self.ssdm = SSDMobileNet(model_file, label_file, num_threads=num_threads, edgetpu=edgetpu, libedgetpu=libedgetpu, score_threshold=score_threshold)
    if wanted_labels is None:
      wanted_labels = ['person']
    self.wanted_labels = wanted_labels
    self.score_threshold = score_threshold
    self.labels = self.ssdm.labels

  def detect_image(self, img):
    #t0 = time.time()
    inp = self.ssdm.prepare_image(img)
    boxes, labels, scores = self.ssdm.predict(inp,original_image_size=img.size)
    #t1 = time.time()
    #print("detect_time={}".format(t1 - t0))
    return_boxs = []
    return_lbls = []
    return_scrs = []
    for i in range(len(boxes)):
      if labels[i] in self.wanted_labels and scores[i] >= self.score_threshold:
        box = boxes[i]
        return_boxs.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        return_lbls.append(labels[i])
        return_scrs.append(scores[i])
    return (return_boxs, return_lbls, return_scrs)

if __name__ == "__main__":
  test()

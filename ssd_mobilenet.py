import time
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageColor
import numpy as np
#from tflite_runtime.interpreter import Interpreter
import tensorflow as tf

class SSDMobileNet:
  def __init__(self,model_path,label_path,num_threads=None):
    self.interpreter = tf.lite.Interpreter(model_path)
    if num_threads is not None:
        self.interpreter.set_num_threads(num_threads)
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

    indices = np.where(output[2] >= confidence)
    reorder = [1,0,3,2]
    if original_image_size is not None:
      w, h = original_image_size
    else:
      w, h = (self.width, self.height)
    boxes = output[0][indices][:,reorder] * [w,h,w,h]
    # import pdb; pdb.set_trace()
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
  def __init__(self, wanted_label=None, model_file=None, label_file=None, num_threads=None):
    if model_file is None:
        model_file = 'ssd_mobilenet.tflite'
    if label_file is None:
        label_file = 'coco_labelmap.txt'
    self.ssdm = SSDMobileNet(model_file, label_file, num_threads=num_threads)
    self.wanted_label = wanted_label
  def detect_image(self, img):
    #t0 = time.time()
    inp = self.ssdm.prepare_image(img)
    boxes, labels, scores = self.ssdm.predict(inp,original_image_size=img.size)
    #t1 = time.time()
    #print("detect_time={}".format(t1 - t0))
    res = []
    for i in range(len(boxes)):
      if self.wanted_label is None or labels[i] == self.wanted_label:
        box = boxes[i]
        res.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
    return res

if __name__ == "__main__":
  test()


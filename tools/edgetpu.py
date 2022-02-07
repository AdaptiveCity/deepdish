from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils

class EDGETPU():
  def __init__(self, wanted_labels=None, model_file=None, label_file=None, num_threads=None, edgetpu=False, libedgetpu=None, score_threshold=0.5):
    if model_file is None:
      model_file = 'detectors/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite'
    if label_file is None:
      label_file = 'detectors/mobilenet/labels.txt'
    self.use_edgetpu = True
    self.num_threads = 1
    self.engine = DetectionEngine(model_file)
    self.labels = dataset_utils.ReadLabelFile(label_file)
    if wanted_labels is None:
      wanted_labels = ['person']
    self.wanted_labels = wanted_labels
    self.score_threshold = score_threshold
    # newer versions of edgetpu library have a new interface
    detect_op = getattr(self.engine, 'detect_with_image', None)
    if callable(detect_op):
      self.detector = detect_op
    else:
      self.detector = self.engine.DetectWithImage

  def detect_image(self, img):
    img = img.convert('RGB')
    ans = self.detector(
        img,
        threshold=self.score_threshold,
        keep_aspect_ratio=False,
        relative_coord=False,
        top_k=20)
    return_boxs = []
    return_lbls = []
    return_scrs = []
    for obj in ans:
      lbl = self.labels[obj.label_id+1]
      box = obj.bounding_box.flatten()
      scr = obj.score
      if lbl in self.wanted_labels and scr >= self.score_threshold:
        return_boxs.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
        return_lbls.append(lbl)
        return_scrs.append(scr)
    return (return_boxs, return_lbls, return_scrs)


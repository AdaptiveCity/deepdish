import sys
import platform
import os
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
from time import time
# pylint: disable=g-import-not-at-top
try:
    import keras
except:
    try:
        from tensorflow import keras
    except:
        pass
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

class YOLOV5:
    def __init__(self, wanted_labels=None, model_file=None, label_file=None, num_threads=None, edgetpu=False, libedgetpu=None, score_threshold=0.25):
        basedir = os.getenv('DEEPDISHHOME','.')
        if model_file is None:
          model_file = os.path.join(basedir, 'detectors/yolov5/yolov5s-int8.tflite')
        if label_file is None:
          label_file = os.path.join(basedir, 'detectors/yolov5/coco_classes.txt')
        self.cfg_file = os.path.join(basedir, 'detectors/yolov5/yolov5s.yaml')
        if wanted_labels is None:
          wanted_labels = ['person']
        self.wanted_labels = wanted_labels
        self.label_file = label_file
        self.score_threshold = score_threshold
        self.labels = self._get_labels()
        self.use_edgetpu = edgetpu
        self.int8 = False

        if 'saved_model' in model_file:
            self.mode = 'saved_model'
            if 'keras' not in sys.modules:
                print('yolov5: saved_model mode requires keras')
                sys.exit(1)
        elif '.tflite' in model_file:
            self.mode = 'tflite'
            if 'int8' in model_file: self.int8 = True
        else:
            print('unable to determine format of yolov5 model')
            sys.exit(1)

        if libedgetpu is None:
            libedgetpu = edgetpu_lib_name()

        if self.mode == 'tflite':
            # Load TFLite model and allocate tensors.
            self.interpreter = Interpreter(
                model_path=model_file,
                num_threads=num_threads,
                experimental_delegates=[load_delegate(libedgetpu)] if self.use_edgetpu else None)
            self.interpreter.allocate_tensors()
            self.num_threads = num_threads
            # Get input and output tensors.
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            _, self.height, self.width, _ = self.input_details[0]['shape'].tolist()
        elif self.mode == 'saved_model':
            self.model = keras.models.load_model(model_file)
            self.num_threads = 1
            _, self.height, self.width, _ = self.model.inputs[0].shape.as_list()

        yaml_file = Path(self.cfg_file)
        with open(yaml_file) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        self.anchors = cfg['anchors']

    def _get_labels(self):
        labels_path = os.path.expanduser(self.label_file)
        with open(labels_path) as f:
            labels = {i: line.strip() for i, line in enumerate(f.readlines())}
        return labels

    def detect_image(self, img):
        img_size = img.size
        img_resized = img.convert('RGB').resize((self.width, self.height), Image.ANTIALIAS)
        input_data = np.expand_dims(img_resized, 0).astype(np.float32)

        if self.int8:
            scale, zero_point = self.input_details[0]['quantization']
            input_data = (input_data / scale + zero_point).astype(np.uint8)

        if self.mode == 'tflite':
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            raw = np.copy(output_data)
        elif self.mode == 'saved_model':
            input_data /= 255.0
            output_data = self.model(input_data).numpy()

        if self.int8:
            scale, zero_point = self.output_details[0]['quantization']
            output_data = output_data.astype(np.float32)
            output_data = (output_data - zero_point) * scale

        x = np.copy(output_data)
        boxes = np.copy(x[..., :4])
        boxes[..., 0] = x[..., 0] - x[..., 2] / 2
        boxes[..., 1] = x[..., 1] - x[..., 3] / 2
        boxes[..., 2] = x[..., 0] + x[..., 2] / 2
        boxes[..., 3] = x[..., 1] + x[..., 3] / 2
        x[..., 5:] *= x[..., 4:5]
        best_classes = np.expand_dims(np.argmax(x[..., 5:], axis=-1), axis=-1)
        confidences = np.take_along_axis(x, best_classes + 5, axis=-1)
        y = np.concatenate((boxes, confidences, best_classes.astype(np.float32)), axis=-1)
        y = y[np.where(y[..., 4] >= self.score_threshold)]
        y[...,:4] *= np.array([img_size[0], img_size[1], img_size[0], img_size[1]])

        return_boxs = []
        return_lbls = []
        return_scrs = []

        for *xyxy, score, labelidx in y:
            label=self.labels[int(labelidx)]
            if label in self.wanted_labels and score >= self.score_threshold:
                tlwh = np.copy(xyxy)
                tlwh[2] = xyxy[2] - xyxy[0]
                tlwh[3] = xyxy[3] - xyxy[1]
                return_boxs.append(list(tlwh))
                return_lbls.append(label)
                return_scrs.append(score)
        return (return_boxs, return_lbls, return_scrs)

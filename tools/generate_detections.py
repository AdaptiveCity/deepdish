# vim: expandtab:ts=4:sw=4
from time import time
import os
import errno
import argparse
import numpy as np
import cv2
import cv2
# pylint: disable=g-import-not-at-top
# Be capable of running partial functionality even without all dependencies installed
try:
    import tensorflow as tf
except:
    pass
try:
    # Import TFLite interpreter from tflite_runtime package if it's available.
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    # If not, fallback to use the TFLite interpreter from the full TF package.
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate
# pylint: enable=g-import-not-at-top


def _run_in_batches(f, data_in, out, batch_size):
    data_len = len(out)
    num_batches = int(data_len / batch_size)
    batch_data = np.zeros((batch_size,) + data_in.shape[1:])

    s, e = 0, 0
    for i in range(num_batches):
        s, e = i * batch_size, (i + 1) * batch_size
        batch_data[0:] = data_in[s:e]
        out[s:e] = f(batch_data)
    if e < len(out):
        batch_data[0:(len(out)-e)] = data_in[e:]
        out[e:] = f(batch_data)[0:(len(out)-e)]

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image

class DummyImageEncoder(object):
    def __init__(self):
        self.height, self.width = 16, 8
        self.image_shape = 16, 8, 3
        self.feature_dim = 128

    def __call__(self, data_in, batch_size=32):
        mat = np.array(data_in,dtype=np.float32)

        mat = np.average(mat, axis=3)
        mat = mat.reshape((-1, 128))
        mat = mat - 128
        out = np.zeros(mat.shape,dtype=np.float32)
        for i in range(mat.shape[0]):
            l = np.sqrt(np.sum(mat[i]**2,axis=0))
            if l == 0:
                out[i] = mat[i]
                out[i,0] = 1
            else:
                out[i] = mat[i]/l
        return out

class ConstantImageEncoder(object):
    def __init__(self):
        self.height, self.width = 16, 8
        self.image_shape = 16, 8, 3
        self.feature_dim = 128

    def __call__(self, data_in, batch_size=32):
        out = np.zeros((data_in.shape[0], 128),dtype=np.float32)
        out[:,0] = 1
        return out

class ImageEncoder(object):

    def __init__(self, checkpoint_filename, input_name="images",
                 output_name="features"):
        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.compat.v1.import_graph_def(graph_def, name="net")
            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(open(checkpoint_filename,'rb').read())
        self.concrete_func = wrap_frozen_graph(graph_def,
                                               inputs="net/%s:0" % input_name,
                                               outputs="net/%s:0" % output_name)

        assert len(self.concrete_func.outputs) == 1
        assert self.concrete_func.outputs[0].shape.rank == 2
        assert len(self.concrete_func.inputs) == 1
        assert self.concrete_func.inputs[0].shape.rank == 4
        self.feature_dim = self.concrete_func.outputs[0].shape.as_list()[-1]
        self.image_shape = self.concrete_func.inputs[0].shape.as_list()[1:]
        self.height, self.width, _ = self.image_shape

    def __call__(self, data_in, batch_size=32):
        out = np.zeros((len(data_in), self.feature_dim), np.float32)
        _run_in_batches(self.concrete_func, tf.identity(data_in), out, batch_size)
        return out

# Run the image encoder using TFLite only
class TFLiteImageEncoder(object):
    def __init__(self, tflite_filename, input_name=None, output_name=None, num_threads=1):
        self.interpreter = Interpreter(model_path=tflite_filename, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.input_tensor_index = self.input_detail['index']
        self.output_tensor_index = self.output_detail['index']
        self.image_shape = self.input_detail['shape'][1:]
        self.height, self.width, _ = self.image_shape.tolist()
        self.feature_dim = self.output_detail['shape'][1]
        self.max_batch_size = self.output_detail['shape'][0]

    def __call__(self, data_in, batch_size=1):
        out = np.zeros((len(data_in), self.feature_dim), np.float32)

        def _internal_fn(patches):
            patches2 = np.array(patches).astype(np.float32)
            self.interpreter.set_tensor(self.input_tensor_index, patches2)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_tensor_index)

        if self.max_batch_size and batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        _run_in_batches(_internal_fn, data_in, out, batch_size)
        return out


def create_box_encoder(model_filename, input_name="images",
                       output_name="features", batch_size=32, num_threads=1):
    if 'dummy' in model_filename:
        image_encoder = DummyImageEncoder()
    if 'constant' in model_filename:
        image_encoder = ConstantImageEncoder()
    elif 'tflite' in model_filename:
        image_encoder = TFLiteImageEncoder(model_filename, input_name, output_name, num_threads=num_threads)
    else:
        image_encoder = ImageEncoder(model_filename, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes, timing=False):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        t1 = time()
        result = image_encoder(image_patches, batch_size)
        t2 = time()
        if timing:
            return result, t2 - t1
        else:
            return result

    encoder.image_encoder = image_encoder
    encoder.width, encoder.height = image_encoder.width, image_encoder.height
    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()

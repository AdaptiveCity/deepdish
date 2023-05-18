# DeepDish

Object recognition, tracking and counting (work-in-progress)

DeepDish is a CNN-based sensor designed to track and count people crossing a 'countline' assigned to the
camera field-of-view, using WiFi for real-time reporting to the Adaptive City platorm. The sensor uses a Raspberry
Pi and a Python framework with multiple alternative CNN models such that relative performance in terms of speed,
accuracy and energy consumption can be assessed.


## Installation

Use of the Docker container is recommended for now. For x86-64 workstations with docker support for GPUs (tested with docker 20.10.16):
```
docker pull wp274/deepdish
./run.sh python3 deepdish.py <options>
```
If you want to build the docker image yourself then run `make docker` and edit `run.sh` to set `IMAGE=deepdish`.

For Raspberry Pi the docker image `mrdanish/deepdish-rpi-tflite-armv7`
is available and a sample script `pi-tflite-run.sh` is provided in
this repository to run it, just like `run.sh` shown above. The
[Hypriot](https://blog.hypriot.com/) distribution of Linux is
recommended because it comes pre-installed with docker for Raspberry
Pi.

## Simple examples (on x86-64 workstation)

Use the SSD MobileNet backend with v1.
- `./run.sh python3 deepdish.py --model detectors/mobilenet/ssdmobilenetv1.tflite --labels detectors/mobilenet/labels.txt --encoder-model encoders/mars-64x32x3.tflite --input input_file.mp4 --output output_file.mp4`

Face detection options include --facemodel (Yunet, DNN, Haar, MTCNN), --trackermodel (KCF, KLT, CAMSHIFT), --motmodel (SORT, CENTROID), --imgtmodel (BOX, GAUSSIAN, DP), --dodraw (TRUE, FALSE), --outputfaces (output_file.csv)
Additional tunable options include --skipfaces (detection/tracker interval), --facescore (confidence interval to use for face detection)
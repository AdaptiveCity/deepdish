# DeepDish

Object recognition, tracking and counting (work-in-progress)

![person detection and tracking](docs/images/titan.png)

DeepDish is a CNN-based sensor designed to track and count people crossing a 'countline' assigned to the
camera field-of-view, using WiFi for real-time reporting to the Adaptive City platorm. The sensor uses a Raspberry
Pi and a Python framework with multiple alternative CNN models such that relative performance in terms of speed, 
accuracy and energy consumption can be assessed.

## Installation

Use of the Docker container is recommended for now.
```
make docker
./run.sh python3 deepdish.py <options>
```
## Overview

The Raspberry Pi and camera have been mounted into a custom housing as below:

![prototype DeepDish unit](docs/images/huey.png)

The basic internal data pipeline is:

![pipeline](docs/images/tracking-by-detection-pipeline.png)

## YOLO model

Be sure to download the H5 file to the correct location.

- `wget -O detectors/yolo/yolo.h5 https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5`

## Simple examples

Use the SSD MobileNet backend with v1.
- `./run.sh python3 deepdish.py --model detectors/mobilenet/ssdmobilenetv1.tflite --labels detectors/mobilenet/labels.txt --encoder-model encoders/mars-64x32x3.pb --input input_file.mp4 --output output_file.mp4`

Use the Yolo backend.
- `./run.sh python3 deepdish.py --model detectors/yolo --labels detectors/yolo --encoder-model encoders/mars-64x32x3.pb --input input_file.mp4 --output output_file.mp4`

Use the EdgeTPU backend with one of the SSD MobileNet v2 models and track objects identified as cars, buses, trucks or bicycles:
- `./run.sh python3 deepdish.py --edgetpu --model=detectors/mobilenet/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite --labels=detectors/mobilenet/labels.txt --encoder-model=encoders/mars-64x32x3.pb --wanted-labels car,bus,truck,bicycle`

## MQTT output examples

`acp_ts` is a Python timestamp, `acp_id` is an identifier as configured on the command-line, `acp_event` tells us if something triggered the message (such as a crossing) or if not present then it is a heartbeat message, `acp_event_value` is a parameter for the event (in the case of crossing, whether the direction was towards the negative or positive side of the line). The other parameters are indexed by category, ergo if you are counting people then `*_person` are the relevant statistics.

* `{"acp_ts": "1606480244.4554827", "acp_id": "deepdish-dd01", "acp_event": "crossing", "acp_event_value": "neg", "temp": 61.835, "poscount_person": 5, "negcount_person": 7, "diff_person": -2, "intcount_person": 12, "delcount_person": 1}`
  * Someone walked from the positive side to the negative side (by default, from right to left as viewed on the output if displayed). So far, people have gone five times right and seven times left. Some convenience calculations are the difference: 5 - 7 = -2, and the total number of intersections with the counting line: 5 + 7 = 12. One tracking identity has been deleted so far (due to not appearing for a certain number of frames).

* `{"acp_ts": "1606480245.8179724", "acp_id": "deepdish-dd01", "acp_event": "crossing", "acp_event_value": "pos", "temp": 62.322, "poscount_person": 6, "negcount_person": 7, "diff_person": -1, "intcount_person": 13, "delcount_person": 1}`
  * Someone walked from the negative side to the positive side (by default, from left to right as viewed on the output if displayed). So far, people have gone six times right and seven times left. Some convenience calculations are the difference: 6 - 7 = -1, and the total number of intersections with the counting line: 6 + 7 = 13. Two tracking identities have been deleted so far (due to not appearing for a certain number of frames).

* `{"acp_ts": "1606480354.9866521", "acp_id": "deepdish-dd01", "temp": 58.426, "poscount_person": 6, "negcount_person": 7, "diff_person": -1, "intcount_person": 13, "delcount_person": 2}`
  * Heartbeat pulse. Same status as above.

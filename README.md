DeepDish - object recognition, tracking and counting (work-in-progress)

* Install

Use of the Docker container is recommended for now.

- make docker
- ./run.sh python3 deepdish.py <options>

* Simple examples

- ./run.sh python3 deepdish.py --model detectors/mobilenet/ssdmobilenetv1.tflite --labels detectors/mobilenet/labels.txt --encoder-model encoders/mars-64x32x3.pb --input input_file.mp4 --output output_file.mp4

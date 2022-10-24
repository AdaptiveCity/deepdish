#!/bin/bash

INP="${1:-chall1.mp4}"
OUT="${2:-out_$INP}"

echo "Input: $INP, Output: $OUT"
python3 deepdish.py  --model detectors/mobilenet/ssdmobilenetv1.tflite --labels detectors/mobilenet/labels.txt --encoder-model encoders/mars-64x32x3.tflite --input "$INP" --output "$OUT" \
        --3d --sensor-width-mm 6.69 --sensor-height-mm 5.55 --focallength-mm 3.2 --elevation-m 1 --tilt-deg 80 --roll-deg 0 --topdownview-size-m "5,5"
#         --line '0,240,640,240' \

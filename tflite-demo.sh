#!/bin/bash


INP="${1:-chall1.mp4}"
OUT="${2:-out_$INP}"

DEFAULT_MODEL=detectors/efficientdet_lite0/efficientdet_lite0.tflite
MODEL="${3:-$DEFAULT_MODEL}"

python3 deepdish.py  --model "$MODEL" --encoder-model encoders/mars-64x32x3.pb --input "$INP" --output "$OUT" \
        --wanted-labels 'person,bicycle,car,motorcycle' --disable-background-subtraction # --raw-output
# --3d --sensor-width-mm 6.69 --sensor-height-mm 5.55 --focallength-mm 3.2 --elevation-m 5 --tilt-deg 80 --roll-deg 0 --topdownview-size-m "60,40"

#!/bin/bash

docker build -t mrdanish/deepdish-rpi-tflite-armv7 \
  -f Dockerfile.rpi-tflite-armv7 \
  --build-arg USER_ID=1000 \
  --build-arg GROUP_ID=1000 \
  .

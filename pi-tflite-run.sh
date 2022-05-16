#!/bin/bash

IMAGE=mrdanish/deepdish-rpi-tflite-armv7

docker run --net=host --privileged -it --rm \
        -v "$PWD":/work \
        -w /work \
        --device /dev/video0:/dev/video0 \
        "$IMAGE" $@


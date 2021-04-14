#!/bin/bash

docker build -t mrdanish/deepdish-rpi \
  -f Dockerfile.rpi \
  --build-arg USER_ID=1000 \
  --build-arg GROUP_ID=1000 \
  .

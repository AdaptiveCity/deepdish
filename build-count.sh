#!/bin/bash

docker build -t deep_sort_yolov3_count \
  -f Dockerfile.count \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .
# docker buildx build \
#   --push \
#   --platform linux/amd64 \
#   -t mrdanish/count \
#   --build-arg USER_ID=$(id -u) \
#   --build-arg GROUP_ID=$(id -g) \
#   .

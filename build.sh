#!/bin/bash

docker build -t mrdanish/deepdish \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .

#!/bin/bash

docker run --gpus 1 --net=host -it --rm \
       -e XAUTHORITY=/home/mrd45/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v /local/scratch-2/mrd45/src/deep_sort_yolov3:/tracker \
       -v $PWD:/work \
       -v "$HOME":/home/mrd45:ro \
       -u `id -u`:`id -g` \
       mrdanish/deep_sort_yolov3 $*

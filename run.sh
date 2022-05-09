#!/bin/bash

docker run --gpus 1 --net=host -it --rm \
       --env TF_FORCE_GPU_ALLOW_GROWTH=true \
       -e XAUTHORITY=/home/mrd45/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v $PWD:/work \
       -v "$HOME":"$HOME":ro \
       -u `id -u`:`id -g` \
       mrdanish/deepdish $*

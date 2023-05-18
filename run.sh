#!/bin/bash

IMAGE=mrdanish/deepdish
#IMAGE=deepdish

docker run --net=host -it --rm \
       --env TF_FORCE_GPU_ALLOW_GROWTH=true \
       -e XAUTHORITY=/home/mrd45/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v $PWD:/work \
       -v "$HOME":"$HOME":ro \
       -u `id -u`:`id -g` \
       "$IMAGE" $*

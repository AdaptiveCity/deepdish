#!/bin/bash

docker run --gpus 1 --net=host -it --rm \
       -e XAUTHORITY=/home/mrd45/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v $PWD:/work \
       -v "$HOME":"$HOME":ro \
       -u `id -u`:`id -g` \
       mrdanish/deepdish $*

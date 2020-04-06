FROM tensorflow/tensorflow:1.15.2-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y # && apt-get upgrade -y
RUN apt-get install -y \
            git \
            python3-matplotlib \
            python3-numpy \
            python3-sklearn \
            python3-opencv \
            fonts-freefont-ttf \
            vim less wget

RUN pip3 install --upgrade pip
RUN pip3 install keras quart

RUN mkdir -p /tracker/detectors/yolo

RUN wget -O /tracker/detectors/yolo/yolo.h5 https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN mkdir -p /work
RUN chown -R user:user /work # /yolo

# Allow password-less 'root' login with 'su'
RUN passwd -d root
RUN sed -i 's/nullok_secure/nullok/' /etc/pam.d/common-auth

RUN echo $'#!/bin/bash\nPYTHONPATH=/tracker DEEPSORTHOME=/tracker YOLOHOME=/tracker python /tracker/count.py $*' > /usr/bin/count.sh

RUN chmod +x /usr/bin/count.sh

COPY *.py /tracker/
COPY detectors/* /tracker/detectors/
COPY yolo3/*.py /tracker/yolo3/
COPY tools/*.py /tracker/tools/
COPY deep_sort/*.py /tracker/deep_sort/

USER user

ENV PYTHONPATH=/tracker

WORKDIR /work

CMD /bin/bash


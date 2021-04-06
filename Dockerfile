FROM tensorflow/tensorflow:1.15.2-gpu-py3
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install -y --allow-downgrades \
            git \
            python3-matplotlib \
            python3-numpy \
            python3-sklearn \
            python3-opencv \
            fonts-freefont-ttf \
            vim less wget \
            libcudnn7=7.6.5.32-1+cuda10.0 # force install of cuda10.0 compatible package
RUN pip3 install --upgrade pip
RUN pip3 install keras==2.3.1 quart gmqtt cameratransform scipy==1.1.0 uvloop
RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

USER root
RUN mkdir -p /deepdish/detectors/yolo
RUN wget -O /deepdish/detectors/yolo/yolo.h5 https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN mkdir -p /work
RUN chown -R user:user /work # /yolo

# Allow password-less 'root' login with 'su'
RUN passwd -d root
RUN sed -i 's/nullok_secure/nullok/' /etc/pam.d/common-auth

RUN echo $'#!/bin/bash\nPYTHONPATH=/deepdish DEEPDISHHOME=/deepdish python3 /deepdish/deepdish.py $@' > /usr/bin/deepdish
RUN echo $'#!/bin/bash\nPYTHONPATH=/deepdish DEEPDISHHOME=/deepdish python3 /deepdish/deepdish.py --model detectors/mobilenet/ssdmobilenetv1.tflite --labels detectors/mobilenet/labels.txt --encoder-model encoders/mars-64x32x3.pb --input "$1" --output "$2" ${@:3}' > /usr/bin/simple

RUN chmod +x /usr/bin/deepdish /usr/bin/simple

COPY *.py /deepdish/
COPY detectors/mobilenet/* /deepdish/detectors/mobilenet/
COPY detectors/yolo/* /deepdish/detectors/yolo/
COPY encoders/* /deepdish/encoders/
COPY yolo3/*.py /deepdish/yolo3/
COPY tools/*.py /deepdish/tools/
COPY deep_sort/*.py /deepdish/deep_sort/

USER user

ENV PYTHONPATH=/deepdish

WORKDIR /work

CMD /bin/bash

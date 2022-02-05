FROM tensorflow/tensorflow:2.7.0-gpu
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
	    libcudnn8-dev \
	    mesa-common-dev libgl1-mesa-dev libgles2-mesa-dev ocl-icd-opencl-dev libegl1-mesa-dev libgles2-mesa-dev

# These were only installed to pull in non-Python dependencies:
RUN apt-get remove -y \
	    python3-matplotlib \
	    python3-numpy \
	    python3-sklearn \
	    python3-opencv

RUN pip3 install --upgrade pip
RUN pip3 install -U https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl
RUN pip3 install -U keras quart gmqtt cameratransform scipy uvloop==0.14.0 matplotlib opencv-python scikit-learn numpy tflite_support datumaro hypercorn

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
COPY detectors/efficientdet_lite0/* /deepdish/detectors/efficientdet_lite0/
COPY encoders/* /deepdish/encoders/
COPY yolo3/*.py /deepdish/yolo3/
COPY tools/*.py /deepdish/tools/
COPY deep_sort/*.py /deepdish/deep_sort/

USER user

ENV PYTHONPATH=/deepdish

WORKDIR /work

CMD /bin/bash

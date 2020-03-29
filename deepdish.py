#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import


import os
import io
from timeit import time
import warnings
import sys
import argparse
import signal

import numpy as np
import cv2
from PIL import Image

from ssd_mobilenet import SSD_MOBILENET
from intersection import any_intersection, intersection

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import asyncio
import concurrent.futures

from quart import Quart


webapp = Quart(__name__)

@webapp.route('/')
async def hello():
    return 'hello'


class FreshQueue(asyncio.Queue):
    """A subclass of queue that keeps only one, fresh item"""
    def _init(self, maxsize):
        self._queue = []
    def _put(self, item):
        self._queue = [item]
    def _get(self):
        item = self._queue[0]
        self._queue = []
        return item

class Pipeline:
    """Object detection and tracking pipeline"""

    def __init__(self, args, input=None):
        self.args = args

        # Initialise camera & camera viewport
        self.init_camera(input)

        # Initialise object detector (for some reason it has to happen
        # here & not within detect_objects(), or else the inference engine
        # gets upset and starts throwing NaNs at me. Thanks, Python.)
        self.object_detector = SSD_MOBILENET(wanted_label='person', model_file=self.args.model, label_file=self.args.labels, num_threads=self.args.num_threads)

        # Initialise feature encoder
        if self.args.encoder_model is None:
            model_filename = '{}/mars-64x32x3.pb'.format(self.args.deepsorthome)
        else:
            model_filename = self.args.encoder_model

        self.encoder = gdet.create_box_encoder(model_filename,batch_size=self.args.encoder_batch_size)

        # Initialise tracker
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.args.max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric,max_iou_distance=self.args.max_iou_distance, max_age=self.args.max_age)

        # Initialise database
        self.db = {}
        self.delcount = 0

        self.loop = asyncio.get_event_loop()

    def init_camera(self, input):
        if input is None:
            self.input = self.args.input
        else:
            self.input = input
        self.cap = cv2.VideoCapture(self.input)

        # Configure the 'counting line' in the camera viewport
        if self.args.line is None:
            w, h = self.args.camera_width, self.args.camera_height
            self.countline = np.array([[w/2,0],[w/2,h]],dtype=int)
        else:
            self.countline = np.array(list(map(int,self.args.line.strip().split(','))),dtype=int).reshape(2,2)
        self.cameracountline = self.countline.astype(float)

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame

    async def capture(self, q):
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                while True:
                    frame = await self.loop.run_in_executor(pool, self.read_frame)
                    if frame is None:
                        print('Frame is None')
                        break
                    await q.put(frame)
                    await asyncio.sleep(1.0/30.0)
        finally:
            self.cap.release()

    async def detect_objects(self, q_in, q_out):
        # Initialise background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2()

        frameCount = 0
        with concurrent.futures.ThreadPoolExecutor() as pool:
            while True:
                frameCount += 1

                # Obtain next video frame
                frame = await q_in.get()

                if self.args.camera_flip:
                    # If we need to flip the image vertically
                    frame = cv2.flip(frame, 0)

                # Apply background subtraction to find image-mask of areas of motion
                fgMask = backSub.apply(frame)

                # Convert to PIL Image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))

                # Run object detection engine within a Thread Pool
                print(frame)
                boxes0 = await self.loop.run_in_executor(pool, self.object_detector.detect_image, image)

                # Filter object detection boxes, including only those with areas of motion
                boxes = []
                for (x,y,w,h) in boxes0:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    # Check if the box includes any detected motion
                    if np.any(fgMask[x:x+w,y:y+h]):
                        boxes.append((x,y,w,h))

                # Send results to next step in pipeline
                await q_out.put((frame, boxes))

    async def encode_features(self, q_in, q_out):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            while True:
                # Obtain next video frame and object detection boxes
                (frame, boxes) = await q_in.get()

                # Run feature encoder within a Thread Pool
                features = await self.loop.run_in_executor(pool, self.encoder, frame, boxes)

                # Build list of 'Detection' objects and send them to next step in pipeline
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
                await q_out.put(detections)

    async def track_objects(self, q_in, q_out):
        while True:
            detections = await q_in.get()
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.args.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            self.tracker.predict()
            self.tracker.update(detections)
            await q_out.put(detections)

    async def process_results(self, q_in):
        while True:
            detections = await(q_in.get())
            # a = (cameracountline[0,0], cameracountline[0,1])
            # b = (cameracountline[1,0], cameracountline[1,1])
            # drawline([a, b], fill=(0,0,255), width=3)

            for track in self.tracker.deleted_tracks:
                i = track.track_id
                if track.is_deleted():
                    self.check_track(track.track_id)

    def check_track(self, i):
        if i in self.db and len(self.db[i]) > 1:
            if any_intersection(self.cameracountline[0], self.cameracountline[1], np.array(self.db[i])):
                self.delcount+=1
                print("delcount={}".format(self.delcount))
            self.db[i] = []

    async def start(self):
        cameraQueue = FreshQueue()
        objectQueue = asyncio.Queue(maxsize=1)
        detectionQueue = asyncio.Queue(maxsize=1)
        resultQueue = asyncio.Queue(maxsize=1)

        asyncio.ensure_future(self.process_results(resultQueue))
        asyncio.ensure_future(self.track_objects(detectionQueue, resultQueue))
        asyncio.ensure_future(self.encode_features(objectQueue, detectionQueue))
        asyncio.ensure_future(self.detect_objects(cameraQueue, objectQueue))
        await self.capture(cameraQueue)

# MODEL = 'ssdmobilenetv1.tflite'
# LABELS = 'labelmap.txt'
# NUM_THREADS = 1
# MAX_COSINE_DISTANCE = 0.6
# NMS_MAX_OVERLAP = 0.9
# ENCODER_MODEL = 'mars-64x32x3.pb'
# MAX_IOU_DISTANCE = 0.7
# MAX_AGE = 30
# INPUT='mot17-03.mp4'

def get_arguments():
    basedir = os.getenv('DEEPSORTHOME','.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="input MP4 file for video file input",
                        default=None)
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    parser.add_argument('--model', help='File path of object detection .tflite file.',
                        required=True)
    parser.add_argument('--encoder-model', help='File path of feature encoder .pb file.',
                        required=False)
    parser.add_argument('--encoder-batch-size', help='Batch size for feature encoder inference',
                        default=32, type=int, metavar='N')
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    parser.add_argument('--no-framebuf', help='Disable framebuffer display',
                        required=False, action='store_true')
    parser.add_argument('--framebuffer', '-F', help='Framebuffer device',
                        default='/dev/fb0', metavar='DEVICE')
    parser.add_argument('--color-mode', help='Color mode for framebuffer, default: RGBA (see OpenCV docs)',
                        default=None, metavar='MODE')
    parser.add_argument('--max-cosine-distance', help='Max cosine distance', metavar='N',
                        default=0.6, type=float)
    parser.add_argument('--nms-max-overlap', help='Non-Max-Suppression max overlap', metavar='N',
                        default=1.0, type=float)
    parser.add_argument('--max-iou-distance', help='Max Intersection-Over-Union distance',
                        metavar='N', default=0.7, type=float)
    parser.add_argument('--max-age', help='Max age of lost track', metavar='N',
                        default=10, type=int)
    parser.add_argument('--num-threads', '-N', help='Number of threads for tensorflow lite',
                        metavar='N', default=4, type=int)
    parser.add_argument('--deepsorthome', help='Location of model_data directory',
                        metavar='PATH', default=None)
    parser.add_argument('--camera-flip', help='Flip the camera image vertically',
                        default=True, type=bool)
    parser.add_argument('--camera-width', help='Camera resolution width in pixels',
                        default=640, type=int)
    parser.add_argument('--camera-height', help='Camera resolution height in pixels',
                        default=480, type=int)
    parser.add_argument('--streaming', help='Stream video over the web?',
                        default=True, type=bool)
    parser.add_argument('--streaming-port', help='TCP port for web video stream',
                        default=8080, type=int)
    parser.add_argument('--stream-path', help='File to write JPG data into, repeatedly.',
                        default=None)
    parser.add_argument('--control-port', help='UDP port for control console.',
                        default=9090, type=int, metavar='PORT')
    args = parser.parse_args()

    if args.deepsorthome is None:
        args.deepsorthome = basedir

    streamFilename = args.stream_path

    return args

class CommandServer():
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        message = data.decode()
        print('Received %r from %s' % (message, addr))
        print('Send %r to %s' % (message, addr))
        self.transport.sendto(data, addr)

cmdserver = None

@webapp.before_serving
async def main():
    global cmdserver
    loop = asyncio.get_event_loop()
    args = get_arguments()
    pipeline = Pipeline(args)
    cmdserver, protocol = await loop.create_datagram_endpoint(
        lambda: CommandServer(pipeline),
        local_addr=('127.0.0.1', args.control_port))
    asyncio.ensure_future(pipeline.start())

@webapp.after_serving
async def shutdown():
    global cmdserver
    cmdserver.close()

if __name__ == '__main__':
    webapp.run()

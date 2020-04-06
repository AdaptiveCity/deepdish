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
from PIL import ImageDraw
from PIL import ImageFont

from tools.ssd_mobilenet import SSD_MOBILENET
from tools.intersection import any_intersection, intersection

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import asyncio
import concurrent.futures

from quart import Quart, current_app


webapp = Quart(__name__)

@webapp.route('/')
async def hello():
    pipeline = current_app.config.pipeline
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

class FontLib:
    def __init__(self, display_w):
        tinysize = int(24.0 / 640.0 * display_w)
        smallsize = int(40.0 / 640.0 * display_w)
        largesize = int(48.0 / 640.0 * display_w)
        self.table = {'tiny': ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', tinysize),
                      'small': ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', smallsize),
                      'large': ImageFont.truetype('fonts/truetype/freefont/FreeSansBold.ttf', largesize)}
    def fetch(self, name):
        if name in self.table:
            return self.table[name]
        else:
            return self.table['large']

# Details for drawing things on a buffer
class RenderInfo:
    def __init__(self, ratio, fontlib, draw, buffer):
        self.ratio = ratio
        self.fontlib = fontlib
        self.draw = draw
        self.buffer = buffer

##################################################
# Output elements

class FrameInfo:
    def __init__(self, t_frame, count):
        self.t_frame = t_frame
        self.frame_count = count
        self.priority = 0

    def do_text(self, handle, elements):
        handle.write('Frame {}:'.format(self.frame_count))
        for e in elements:
            if isinstance(e, TimingInfo):
                handle.write(' {}={:.0f}ms'.format(e.short_label, e.delta_t*1000))
        handle.write('\n')

class TimingInfo:
    def __init__(self, desc, short_label, delta_t):
        self.description = desc
        self.short_label = short_label
        self.delta_t = delta_t
        self.priority = 1

# A detected object - simply the information conveyed by the object detector
class DetectedObject:
    def __init__(self, bbox):
        self.bbox = bbox
        self.priority = 5
        self.outline = (255, 0, 0)
    def do_render(self, render):
        pts = list(np.int32(np.array(self.bbox).reshape(-1,2) * render.ratio).reshape(-1))
        render.draw.rectangle(pts, outline=self.outline)

# A tracked object based on the output of the tracker
class TrackedObject:
    def __init__(self, bbox, txt):
        self.bbox = bbox
        self.txt = txt
        self.priority = 6
        self.outline = (255, 255, 255)
        self.font_fill = (0, 255, 0)
        self.font = 'small'
    def do_render(self, render):
        pts = list(np.int32(np.array(self.bbox).reshape(-1,2) * render.ratio).reshape(-1))
        render.draw.rectangle(pts, outline=self.outline)
        render.draw.text(self.bbox[:2],str(self.txt), fill=self.font_fill, font=render.fontlib.fetch(self.font))

# Base class for graphical elements that draw a line
class Line:
    def do_render(self, render):
        pts = list(np.int32(np.array(self.pts).reshape(-1,2) * render.ratio).reshape(-1))
        render.draw.line(pts, fill=self.fill, width=self.width)

class TrackedPath(Line):
    def __init__(self, pts):
        self.pts = pts
        self.priority = 3
        self.width = 3
        self.fill = (255, 0, 255)

class TrackedPathIntersection(Line):
    def __init__(self, pts):
        self.pts = pts
        self.priority = 4
        self.width = 5
        self.fill = (0, 0, 255)

class CameraCountLine(Line):
    def __init__(self, pts):
        self.pts = pts
        self.priority = 2
        self.width = 3
        self.fill = (0, 0, 255)

class CameraImage:
    def __init__(self, image):
        self.image = image
        self.priority = 1

    def do_render(self, render):
        render.buffer.paste(self.image)

class CountingStats:
    def __init__(self, negcount, poscount):
        self.negcount = negcount
        self.poscount = poscount
        self.priority = 10
        self.font_fill_negcount = (255, 0, 0)
        self.font_fill_abscount = (0, 255, 0)
        self.font_fill_poscount = (0, 0, 255)
        self.font = 'large'

    def do_render(self, render):
        font = render.fontlib.fetch(self.font)
        [w, h] = render.buffer.size

        (_, dy) = font.getsize(str(self.negcount))
        render.draw.text((0, h - dy), str(self.negcount), fill=self.font_fill_negcount, font=font)

        abscount = abs(self.negcount-self.poscount)
        (dx, dy) = font.getsize(str(abscount))
        render.draw.text(((w - dx)/2, h - dy), str(abscount), fill=self.font_fill_abscount, font=font)

        (dx, dy) = font.getsize(str(self.poscount))
        render.draw.text((w - dx, h - dy), str(self.poscount), fill=self.font_fill_poscount, font=font)

##################################################

class Pipeline:
    """Object detection and tracking pipeline"""

    def __init__(self, args, input=None):
        self.args = args

        # Initialise camera & camera viewport
        self.init_camera(input)
        # Initialise output
        self.init_output(self.args.output)

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
        self.intcount = 0
        self.poscount = 0
        self.negcount = 0

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

    def init_output(self, output):
        self.color_mode = None # fixme
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        (w, h) = (self.args.camera_width, self.args.camera_height)
        self.backbuf = Image.new("RGBA", (w, h), (0,0,0,0))
        self.draw = ImageDraw.Draw(self.backbuf)
        self.output = cv2.VideoWriter(self.args.output,fourcc, fps, (w, h))

    def read_frame(self):
        ret, frame = self.cap.read()
        return (frame, time.time())

    async def capture(self, q):
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                while self.running:
                    (frame, t_frame) = await self.loop.run_in_executor(pool, self.read_frame)
                    #print(frame)
                    if frame is None:
                        print('Frame is None')
                        break
                    await q.put((frame, t_frame))
                    await asyncio.sleep(1.0/30.0)
        finally:
            self.cap.release()

    def run_object_detector(self, image):
        t1 = time.time()
        boxes = self.object_detector.detect_image(image)
        t2 = time.time()
        return (boxes, t2 - t1)

    async def detect_objects(self, q_in, q_out):
        # Initialise background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2()

        frameCount = 0
        with concurrent.futures.ThreadPoolExecutor() as pool:
            while self.running:
                frameCount += 1

                # Obtain next video frame
                (frame, t_frame) = await q_in.get()

                if self.args.camera_flip:
                    # If we need to flip the image vertically
                    frame = cv2.flip(frame, 0)

                # Apply background subtraction to find image-mask of areas of motion
                fgMask = backSub.apply(frame)

                # Convert to PIL Image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))

                # Run object detection engine within a Thread Pool
                (boxes0, delta_t) = await self.loop.run_in_executor(pool, self.run_object_detector, image)

                # Filter object detection boxes, including only those with areas of motion
                boxes = []
                for (x,y,w,h) in boxes0:
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    # Check if the box includes any detected motion
                    if np.any(fgMask[x:x+w,y:y+h]):
                        boxes.append((x,y,w,h))

                # Send results to next step in pipeline
                elements = [FrameInfo(t_frame, frameCount),
                            CameraImage(image),
                            CameraCountLine(self.cameracountline),
                            TimingInfo('Object detection latency', 'objd', delta_t)]
                await q_out.put((frame, boxes, elements))

    async def encode_features(self, q_in, q_out):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            while self.running:
                # Obtain next video frame and object detection boxes
                (frame, boxes, elements) = await q_in.get()

                # Run feature encoder within a Thread Pool
                features = await self.loop.run_in_executor(pool, self.encoder, frame, boxes)

                # Build list of 'Detection' objects and send them to next step in pipeline
                detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
                await q_out.put((detections, elements))

    async def track_objects(self, q_in, q_out):
        while self.running:
            (detections, elements) = await q_in.get()
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.args.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            self.tracker.predict()
            self.tracker.update(detections)
            await q_out.put((detections, elements))

    async def process_results(self, q_in, q_out):
        while self.running:
            (detections, elements) = await(q_in.get())

            for track in self.tracker.deleted_tracks:
                i = track.track_id
                if track.is_deleted():
                    self.check_deleted_track(track.track_id)

            for track in self.tracker.tracks:
                i = track.track_id
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                if i not in self.db:
                    self.db[i] = []

                bbox = track.to_tlbr()

                # Find the bottom-centre of the bounding box & add it to the tracking database
                bottomCentre = np.array([(bbox[0] + bbox[2]) / 2.0, bbox[3]])
                self.db[i].append(bottomCentre)

                if len(self.db[i]) > 1:
                    # If we have more than one datapoint for this tracked object
                    pts = (np.array(self.db[i]).reshape((-1,1,2))).reshape(-1)
                    elements.append(TrackedPath(pts))

                    p1 = self.cameracountline[0]
                    q1 = self.cameracountline[1]
                    p2 = np.array(self.db[i][-1])
                    q2 = np.array(self.db[i][-2])
                    cp = np.cross(q1 - p1,q2 - p2)
                    if intersection(p1,q1,p2,q2):
                        self.intcount+=1
                        print("track_id={} just intersected camera countline; cross-prod={}; intcount={}".format(i,cp,self.intcount))
                        elements.append(TrackedPathIntersection(pts[-4:]))
                        if cp >= 0:
                            self.poscount+=1
                        else:
                            self.negcount+=1
                        # send_mqtt_msg(frameCapTime)

                elements.append(TrackedObject(bbox, str(track.track_id)))

            for det in detections:
                bbox = det.to_tlbr()
                elements.append(DetectedObject(bbox))

            elements.append(CountingStats(self.negcount, self.poscount))

            await q_out.put(elements)

    def graphical_output(self, render : RenderInfo, elements, output_wh : (int, int)):
        (output_w, output_h) = output_wh

        # Clear screen
        self.draw.rectangle([0, 0, output_w, output_h], fill=0, outline=0)

        # Sort elements by display priority
        elements.sort(key=lambda e: e.priority)

        # Draw elements
        for e in elements:
            if hasattr(e, 'do_render'):
                e.do_render(render)

        # Copy backbuf to output
        backarray = np.array(self.backbuf)
        if self.color_mode is not None:
            outputrgba = cv2.cvtColor(backarray, self.color_mode)
        else:
            outputrgba = backarray
        outputrgb = cv2.cvtColor(outputrgba, cv2.COLOR_RGBA2RGB)
        self.output.write(outputrgb)
        #cv2.imshow('main', outputrgb)

    def text_output(self, handle, elements):
        # Sort elements by priority
        elements.sort(key=lambda e: e.priority)

        for e in elements:
            if hasattr(e, 'do_text'):
                e.do_text(handle, elements)

    async def render_output(self, q_in):
        (output_w, output_h) = (self.args.camera_width, self.args.camera_height)
        ratio = 1 #fixme
        render = RenderInfo(ratio, FontLib(output_w), self.draw, self.backbuf)

        try:
            while self.running:
                elements = await q_in.get()

                self.graphical_output(render, elements, (output_w, output_h))

                for e in elements:
                    if isinstance(e, FrameInfo):
                        t_frame = e.t_frame
                        break
                elements.append(TimingInfo('Overall latency', 'overall', time.time() - t_frame))

                self.text_output(sys.stdout, elements)

                await asyncio.sleep(1.0 / 30.0) # FIXME
        finally:
            self.output.release()

    def check_deleted_track(self, i):
        if i in self.db and len(self.db[i]) > 1:
            if any_intersection(self.cameracountline[0], self.cameracountline[1], np.array(self.db[i])):
                self.delcount+=1
                print("delcount={}".format(self.delcount))
            self.db[i] = []

    async def start(self):
        self.running = True
        cameraQueue = FreshQueue()
        objectQueue = asyncio.Queue(maxsize=1)
        detectionQueue = asyncio.Queue(maxsize=1)
        resultQueue = asyncio.Queue(maxsize=1)
        drawQueue = asyncio.Queue(maxsize=1)

        asyncio.ensure_future(self.render_output(drawQueue))
        asyncio.ensure_future(self.process_results(resultQueue, drawQueue))
        asyncio.ensure_future(self.track_objects(detectionQueue, resultQueue))
        asyncio.ensure_future(self.encode_features(objectQueue, detectionQueue))
        asyncio.ensure_future(self.detect_objects(cameraQueue, objectQueue))
        await self.capture(cameraQueue)
        self.running = False

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
    basedir = os.getenv('DEEPDISHHOME','.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="input MP4 file for video file input",
                        default=None)
    parser.add_argument('--output', help="output file with annotated video frames",
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
                        default=False, type=bool)
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

    def connection_lost(self, exc):
        pass

cmdserver = None

@webapp.before_serving
async def main():
    global cmdserver
    loop = asyncio.get_event_loop()
    args = get_arguments()
    pipeline = Pipeline(args)
    current_app.config.pipeline = pipeline
    cmdserver, protocol = await loop.create_datagram_endpoint(
        lambda: CommandServer(pipeline),
        local_addr=('127.0.0.1', args.control_port))
    await pipeline.start()
    for p in asyncio.Task.all_tasks():
        p.cancel()
    # await loop.run_until_complete(asyncio.Task.all_tasks())

@webapp.after_serving
async def shutdown():
    global cmdserver
    cmdserver.close()

if __name__ == '__main__':
    try:
        webapp.run()
    except concurrent.futures.CancelledError:
        pass

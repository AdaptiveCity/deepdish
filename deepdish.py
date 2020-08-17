#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import re
import io
from timeit import time
from time import time
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
from tools.yolo import YOLO
from tools.edgetpu import EDGETPU
from tools.intersection import any_intersection, intersection

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

import asyncio
import uvloop
import aiofiles
import concurrent.futures
from gmqtt import Client as MQTTClient
import json

from quart import Quart, Response, current_app
import threading

class MBox:
    def __init__(self):
        self.message = None
        self.lock = threading.Lock()

    def get_message(self):
        self.lock.acquire()
        message = self.message
        self.lock.release()
        return message

    def set_message(self, message):
        self.lock.acquire()
        self.message = message
        self.lock.release()

def capthread_f(cap, box):
    try:
        prev_t = time()
        ret = True
        while ret:
            t1 = time()
            ret, frame = cap.read()
            if not ret:
              frame = None
            t2 = time()
            #print('{:.02f}ms'.format((t2-prev_t)*1000))
            prev_t = t2
            box.set_message((frame,t2,t2-t1))
    finally:
        cap.release()

class Error(Exception):
    def __init__(self, msg):
        self.message = msg

webapp = Quart(__name__)

class StreamingInfo:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.frame = None
    async def get_frame(self):
        async with self.lock:
            return self.frame
    async def set_frame(self, frame):
        async with self.lock:
            self.frame = frame

streaminfo = StreamingInfo()

async def generate(si):
    # loop over frames from the output stream
    while True:
        await asyncio.sleep(0.003) #FIXME: is this necessary?
        # wait until the lock is acquired
        frame = await si.get_frame()
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if frame is None:
            continue

        t1=time()
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        t2=time()
        #print("imencode={:.0f}ms".format((t2-t1)*1000))

        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        t1=time()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) + b'\r\n')
        t2=time()
        #print("yield={:.0f}ms".format((t2-t1)*1000))

@webapp.route("/")
async def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(streaminfo), mimetype = "multipart/x-mixed-replace; boundary=frame")


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
    def full(self):
        return False

class FontLib:
    def __init__(self, display_w, fontbasedirs = ['.', '/usr/local/share', '/usr/share']):
        tinysize = int(24.0 / 640.0 * display_w)
        smallsize = int(40.0 / 640.0 * display_w)
        largesize = int(48.0 / 640.0 * display_w)

        fontfile = None
        for bd in fontbasedirs:
            f = os.path.join(bd, 'fonts/truetype/freefont/FreeSansBold.ttf')
            if os.path.exists(f):
                fontfile = f
                break
        self.table = {'tiny': ImageFont.truetype(fontfile, tinysize),
                      'small': ImageFont.truetype(fontfile, smallsize),
                      'large': ImageFont.truetype(fontfile, largesize)}
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
        self.font = 'tiny'
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

class FGMask:
    def __init__(self, fgMask):
        self.fgMask = fgMask
        self.priority = 2

    def do_render(self, render):
        image = Image.fromarray(self.fgMask)
        render.buffer.paste(image)

class CountingStats:
    def __init__(self, negcount, poscount):
        self.negcount = negcount
        self.poscount = poscount
        self.priority = 10
        self.font_fill_negcount = (255, 0, 0)
        self.font_fill_abscount = (0, 255, 0)
        self.font_fill_poscount = (0, 0, 255)
        self.font = 'tiny'
        self.labels = list(negcount.keys())
        self.labels.reverse()

    def do_render(self, render):
        font = render.fontlib.fetch(self.font)
        [w, h] = render.buffer.size

        cursor = h
        for lbl in self.labels:
            (_, dy) = font.getsize(str(self.negcount[lbl]))
            cursor -= dy

            render.draw.text((0, cursor), str(self.negcount[lbl]), fill=self.font_fill_negcount, font=font)

            #central = str(abs(self.negcount[lbl]-self.poscount[lbl]))
            central = lbl
            (dx, dy) = font.getsize(central)
            render.draw.text(((w - dx)/2, cursor), central, fill=self.font_fill_abscount, font=font)

            (dx, dy) = font.getsize(str(self.poscount[lbl]))
            render.draw.text((w - dx, cursor), str(self.poscount[lbl]), fill=self.font_fill_poscount, font=font)

##################################################

class Pipeline:
    """Object detection and tracking pipeline"""

    def __init__(self, args):
        self.args = args

        # Initialise camera & camera viewport
        self.init_camera()
        # Initialise output
        self.init_output(self.args.output)

        # Process comma-separated list of wanted labels
        self.wanted_labels = self.args.wanted_labels.strip().split(',')

        # Initialise object detector (for some reason it has to happen
        # here & not within detect_objects(), or else the inference engine
        # gets upset and starts throwing NaNs at me. Thanks, Python.)
        if 'yolo' in self.args.model:
            self.object_detector = YOLO(wanted_labels=self.wanted_labels, model_file=self.args.model, label_file=self.args.labels, num_threads=self.args.num_threads)
        elif self.args.edgetpu:
            self.object_detector = EDGETPU(wanted_labels=self.wanted_labels, model_file=self.args.model, label_file=self.args.labels,
                    num_threads=self.args.num_threads, edgetpu=self.args.edgetpu)
        else:
            self.object_detector = SSD_MOBILENET(wanted_labels=self.wanted_labels, model_file=self.args.model, label_file=self.args.labels,
                    num_threads=self.args.num_threads, edgetpu=self.args.edgetpu)

        # Initialise feature encoder
        if self.args.encoder_model is None:
            model_filename = '{}/mars-64x32x3.pb'.format(self.args.deepsorthome)
        else:
            model_filename = self.args.encoder_model

        self.encoder = gdet.create_box_encoder(model_filename,batch_size=self.args.encoder_batch_size)

        self.background_subtraction = not self.args.disable_background_subtraction

        # Initialise tracker
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.args.max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric,max_iou_distance=self.args.max_iou_distance, max_age=self.args.max_age)

        # Initialise database
        self.db = {}
        self.delcount = dict([(lbl, 0) for lbl in self.wanted_labels])
        self.intcount = dict([(lbl, 0) for lbl in self.wanted_labels])
        self.poscount = dict([(lbl, 0) for lbl in self.wanted_labels])
        self.negcount = dict([(lbl, 0) for lbl in self.wanted_labels])

        self.mqtt = None
        self.topic = self.args.mqtt_topic
        self.mqtt_acp_id = self.args.mqtt_acp_id
        self.heartbeat_delay_secs = self.args.heartbeat_delay_secs

        self.log = self.args.log
        if self.log is not None:
            with open(self.log, mode='w+') as f:
                f.truncate()
        self.loop = asyncio.get_event_loop()

    async def init_mqtt(self):
        if self.args.mqtt_broker is not None:
            self.mqtt = MQTTClient('deepdish')
            if self.args.mqtt_user is not None:
                self.mqtt.set_auth_credentials(self.args.mqtt_user, self.args.mqtt_pass)
            await self.mqtt.connect(self.args.mqtt_broker)
            if self.topic is None:
                self.topic = 'default/topic'

    def init_camera(self):
        self.input = self.args.input
        if self.input is None:
            if self.args.gstreamer is not None:
                src = self.args.gstreamer
            elif self.args.gstreamer_nvidia:
                w, h = self.args.camera_width, self.args.camera_height
                src = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true".format(w,h)
            else:
                src = self.args.camera

            self.input = src
            # Allow live camera frames to be dropped
            self.everyframe = None
        else:
            # Capture every frame from the video file
            self.everyframe = asyncio.Event()
        self.cap = cv2.VideoCapture(self.input)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
        if not self.args.framebuffer:
            self.framebufdev = None
        else:
            self.framebufdev = self.args.framebuffer_device
            fbX = self.framebufdev[-3:]

            vsizefile = '/sys/class/graphics/{}/virtual_size'.format(fbX)
            if not os.path.exists(self.framebufdev) or not os.path.exists(vsizefile):
                #raise Error('Invalid framebuffer device: {}'.format(self.framebufdev))
                print('Invalid framebuffer device: {}'.format(self.framebufdev))
                self.framebufdev = None

        if self.framebufdev is not None:
            (w, h) = (self.args.framebuffer_width, self.args.framebuffer_height)
            if w is None or h is None:
                nums = re.findall('(.*),(.*)', open(vsizefile).read())[0]
                if w is None:
                    w = int(nums[0])
                if h is None:
                    h = int(nums[1])
            self.framebufres = (w, h)

    def shutdown(self):
        self.running = False
        for p in asyncio.Task.all_tasks():
            p.cancel()

    def read_frame_from_box(self, box):
        msg = None
        while msg is None:
            msg = box.get_message()
        (frame, t_frame, dt_cap) = msg
        return (frame, t_frame, dt_cap)

    async def capture(self, q, box):
        try:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                while self.running:
                    frame = None
                    # Fetch next frame
                    (frame, t_frame, dt_cap) = await self.loop.run_in_executor(pool, self.read_frame_from_box, box)
                    if frame is None:
                        print('No more frames.')
                        self.shutdown()
                        break

                    if self.args.camera_flip:
                        # If we need to flip the image vertically
                        frame = cv2.flip(frame, 0)
                    # Ensure frame is proper size
                    frame = cv2.resize(frame, (self.args.camera_width, self.args.camera_height))

                    await q.put((frame, t_frame, time()))

                    # If we are ensuring every frame is processed then wait for
                    # synchronising event to be triggered
                    if self.everyframe is not None:
                        await self.everyframe.wait()
                        self.everyframe.clear()

        finally:
            self.cap.release()

    def run_object_detector(self, image):
        t1 = time()
        (boxes, labels, scores) = self.object_detector.detect_image(image)
        t2 = time()
        return (boxes, labels, scores, t2 - t1)

    async def detect_objects(self, q_in, q_out):
        # Initialise background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2()

        frameCount = 0
        with concurrent.futures.ThreadPoolExecutor() as pool:
            while self.running:
                frameCount += 1

                # Obtain next video frame
                (frame, t_frame, t_prev) = await q_in.get()

                t_frame_recv = time()
                # Apply background subtraction to find image-mask of areas of motion
                if self.background_subtraction:
                    fgMask = backSub.apply(frame)
                    if self.args.enable_background_masking:
                        frame = cv2.bitwise_and(frame,frame,mask = fgMask)
                # Convert to PIL Image
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))
                t_backsub = time()

                # Run object detection engine within a Thread Pool
                (boxes0, labels0, scores0, delta_t) = await self.loop.run_in_executor(pool, self.run_object_detector, image)

                # Filter object detection boxes, including only those with areas of motion
                t1 = time()
                boxes = []
                labels = []
                scores = []
                max_x, max_y = self.args.camera_width, self.args.camera_height
                for ((x,y,w,h), lbl, scr) in zip(boxes0, labels0, scores0):
                    if np.any(np.isnan(boxes0)):
                        # Drop any rubbish results
                        continue
                    x, y = int(np.clip(x,0,max_x)), int(np.clip(y,0,max_y))
                    w, h = int(np.clip(w,0,max_x-x)), int(np.clip(h,0,max_y-y))
                    # Check if the box is almost as large as the camera viewport
                    if w * h > 0.9 * max_x * max_y:
                        # reject as spurious
                        continue
                    # Check if the box includes sufficient detected motion
                    if not self.background_subtraction or np.count_nonzero(fgMask[y:y+h,x:x+w]) >= self.args.background_subtraction_ratio * w * h:
                        boxes.append((x,y,w,h))
                        labels.append(lbl)
                        scores.append(scr)
                t2 = time()

                # Send results to next step in pipeline
                elements = [FrameInfo(t_frame, frameCount),
                            CameraImage(image),
                            CameraCountLine(self.cameracountline),
                            TimingInfo('Frame processing latency', 'fram', t_prev - t_frame),
                            TimingInfo('Frame / Q1 item received latency', 'q1', t_frame_recv - t_prev),
                            TimingInfo('Background subtraction latency', 'bsub', t_backsub - t_frame_recv),
                            TimingInfo('Object detection latency', 'objd', delta_t+(t2-t1))]
                await q_out.put((frame, boxes, labels, scores, elements, time()))

    async def encode_features(self, q_in, q_out):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            while self.running:
                # Obtain next video frame and object detection boxes
                (frame, boxes, labels, scores, elements, t_prev) = await q_in.get()

                t1 = time()
                # Run non-max suppression to eliminate spurious boxes
                boxesA0 = np.array(boxes)
                scoresA0 = np.array(scores)
                indices = preprocessing.non_max_suppression(boxesA0, self.args.nms_max_overlap, scoresA0)
                boxesA1 = boxesA0[indices]
                scoresA1 = scoresA0[indices]
                labels1 = [labels[i] for i in indices]

                # Run feature encoder within a Thread Pool
                features = await self.loop.run_in_executor(pool, self.encoder, frame, boxesA1)
                t2 = time()

                # Build list of 'Detection' objects and send them to next step in pipeline
                detections = [Detection(bbox, lbl, scr, feature) for bbox, lbl, scr, feature in zip(boxesA1, labels1, scoresA1, features)]
                elements.append(TimingInfo('Q1 / Q2 latency', 'q2', (t1 - t_prev)))
                elements.append(TimingInfo('Feature encoder latency', 'feat', (t2-t1)))
                await q_out.put((detections, elements, time()))

    async def track_objects(self, q_in, q_out):
        while self.running:
            (detections, elements, t_prev) = await q_in.get()
            t1 = time()
            self.tracker.predict()
            self.tracker.update(detections)
            t2 = time()
            elements.append(TimingInfo('Q2 / Q3 latency', 'q3', (t1 - t_prev)))
            elements.append(TimingInfo('Tracker latency', 'trak', (t2-t1)))
            await q_out.put((detections, elements, time()))

    async def process_results(self, q_in, q_out):
        while self.running:
            (detections, elements, t_prev) = await(q_in.get())

            t1=time()
            for track in self.tracker.deleted_tracks:
                i = track.track_id
                if track.is_deleted():
                    self.check_deleted_track(track)

            for track in self.tracker.tracks:
                i = track.track_id
                lbl = track.get_label()
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
                        self.intcount[lbl]+=1
                        print("track_id={} ({}) just intersected camera countline; cross-prod={}; intcount={}".format(i,lbl,cp,self.intcount))
                        elements.append(TrackedPathIntersection(pts[-4:]))
                        if cp >= 0:
                            self.poscount[lbl]+=1
                            crossing_type = 'pos'
                        else:
                            self.negcount[lbl]+=1
                            crossing_type = 'neg'
                        await self.publish_crossing_event_to_mqtt(elements, crossing_type)
                        await self.publish_crossing_event_to_log(elements)

                if self.args.object_annotation.lower() == 'id':
                    annot = str(track.track_id)
                elif self.args.object_annotation.lower() == 'label':
                    annot = lbl
                else:
                    annot = ''
                elements.append(TrackedObject(bbox, annot))

            for det in detections:
                bbox = det.to_tlbr()
                elements.append(DetectedObject(bbox))

            elements.append(CountingStats(self.negcount, self.poscount))
            t2=time()
            elements.append(TimingInfo('Q3 / Q4 latency', 'q4', (t1-t_prev)))
            elements.append(TimingInfo('Results processing latency', 'proc', (t2-t1)))

            await q_out.put((elements,time()))

    async def publish_crossing_event_to_mqtt(self, elements, crossing_type):
        if self.mqtt is not None:
            for e in elements:
                if isinstance(e, FrameInfo):
                    t_frame = e.t_frame
                    break
            payload = {'acp_ts': str(t_frame), 'acp_id': self.mqtt_acp_id, 'acp_event': 'crossing', 'acp_event_value': crossing_type}
            for lbl in self.wanted_labels:
                payload.update(dict([('poscount_'+lbl, self.poscount[lbl]), ('negcount_'+lbl, self.negcount[lbl]), ('diff_'+lbl, self.poscount[lbl] - self.negcount[lbl])]))

            self.mqtt.publish(self.topic, json.dumps(payload))

    async def publish_crossing_event_to_log(self, elements):
        if self.log is not None:
            for e in elements:
                if isinstance(e, FrameInfo):
                    count = e.frame_count
                    break
            payload = {'frame_count': count}
            for lbl in self.wanted_labels:
                payload.update(dict([('poscount_'+lbl, self.poscount[lbl]), ('negcount_'+lbl, self.negcount[lbl]), ('diff_'+lbl, self.poscount[lbl] - self.negcount[lbl])]))

            async with aiofiles.open(self.log, mode='a+') as f:
                await f.write(json.dumps(payload) + '\n')

    async def periodic_mqtt_heartbeat(self):
        if self.mqtt is not None:
            while True:
                payload = {'acp_ts': str(time()), 'acp_id': self.mqtt_acp_id}
                for lbl in self.wanted_labels:
                    payload.update(dict([('poscount_'+lbl, self.poscount[lbl]), ('negcount_'+lbl, self.negcount[lbl]), ('diff_'+lbl, self.poscount[lbl] - self.negcount[lbl])]))
                self.mqtt.publish(self.topic, json.dumps(payload))
                await asyncio.sleep(self.heartbeat_delay_secs)

    async def graphical_output(self, render : RenderInfo, elements, output_wh : (int, int)):
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
            outputbgra = cv2.cvtColor(backarray, self.color_mode)
        else:
            outputbgra = backarray
        outputrgb = cv2.cvtColor(outputbgra, cv2.COLOR_BGRA2RGB)
        if self.output is not None:
            self.output.write(outputrgb)
        if self.framebufdev is not None:
            outputrgba = cv2.cvtColor(outputbgra, cv2.COLOR_BGRA2RGBA)
            outputfbuf = cv2.resize(outputrgba, self.framebufres)
            try:
                with open(self.framebufdev, 'wb') as buf:
                    buf.write(outputfbuf)
            except:
                print('failed to write to framebuffer device {} ...disabling it.'.format(self.framebufdev))
                self.framebufdev = None
        await streaminfo.set_frame(outputrgb)

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
                (elements, t_prev) = await q_in.get()

                t1 = time()
                await self.graphical_output(render, elements, (output_w, output_h))

                for e in elements:
                    if isinstance(e, FrameInfo):
                        t_frame = e.t_frame
                        break
                elements.append(TimingInfo('Q4 / Q5 latency', 'q5', t1 - t_prev))
                elements.append(TimingInfo('Graphical display latency', 'disp', time() - t1))
                t_sum = 0
                for e in elements:
                    if isinstance(e, TimingInfo):
                        t_sum += e.delta_t
                elements.append(TimingInfo('Latency sum', 'sum', t_sum))
                t_e2e = time() - t_frame
                elements.append(TimingInfo('End to end latency', 'e2e', t_e2e))
                elements.append(TimingInfo('Missing', 'miss', t_e2e - t_sum))

                self.text_output(sys.stdout, elements)

                if self.everyframe is not None:
                    # Notify other side that this frame is completely processed
                    self.everyframe.set()

        finally:
            self.output.release()

    def check_deleted_track(self, track):
        i = track.track_id
        if i in self.db and len(self.db[i]) > 1:
            if any_intersection(self.cameracountline[0], self.cameracountline[1], np.array(self.db[i])):
                l = track.get_label()
                self.delcount[l]+=1
                print("delcount[{}]={}".format(l,self.delcount[l]))
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

        box = MBox()
        capthread = threading.Thread(target=capthread_f, args=(self.cap,box), daemon=True)
        capthread.start()
        await self.capture(cameraQueue, box)
        self.running = False

def get_arguments():
    basedir = os.getenv('DEEPDISHHOME','.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', help="camera number for live input (OpenCV numbering)",
                        metavar='N', default=0, type=int)
    parser.add_argument('--gstreamer', help='gstreamer pipeline for camera input (instead of camera number)',
                        metavar='PIPELINE', default=None)
    parser.add_argument('--gstreamer-nvidia', help='use nvidia-default gstreamer pipeline (instead of camera number)',
                        action='store_true', default=False)
    parser.add_argument('--input', help="input MP4 file for video file input (instead of camera)",
                        default=None)
    parser.add_argument('--output', help="output file with annotated video frames",
                        default=None)
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    parser.add_argument('--model', help='File path of object detection .tflite file.',
                        required=True)
    parser.add_argument('--edgetpu', help='Enable usage of Edge TPU accelerator.',
                        default=False, action='store_true')
    parser.add_argument('--encoder-model', help='File path of feature encoder .pb file.',
                        required=False)
    parser.add_argument('--encoder-batch-size', help='Batch size for feature encoder inference',
                        default=32, type=int, metavar='N')
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    parser.add_argument('--framebuffer', help='Disable framebuffer display',
                        required=False, action='store_true')
    parser.add_argument('--framebuffer-device', '-F', help='Framebuffer device',
                        default='/dev/fb0', metavar='DEVICE')
    parser.add_argument('--framebuffer-width', help='Framebuffer device resolution (width) override',
                        default=None, metavar='WIDTH')
    parser.add_argument('--framebuffer-height', help='Framebuffer device resolution (height) override',
                        default=None, metavar='HEIGHT')
    parser.add_argument('--color-mode', help='Color mode for framebuffer, default: RGBA (see OpenCV docs)',
                        default=None, metavar='MODE')
    parser.add_argument('--max-cosine-distance', help='Max cosine distance', metavar='N',
                        default=0.2, type=float)
    parser.add_argument('--nms-max-overlap', help='Non-Max-Suppression max overlap', metavar='N',
                        default=0.6, type=float)
    parser.add_argument('--max-iou-distance', help='Max Intersection-Over-Union distance',
                        metavar='N', default=0.7, type=float)
    parser.add_argument('--max-age', help='Max age of lost track (in number of frames)', metavar='N',
                        default=60, type=int)
    parser.add_argument('--wanted-labels', help='Comma-separated list of labels of objects to count',
                        metavar='LABEL1,LABEL2,...', default='person')
    parser.add_argument('--num-threads', '-N', help='Number of threads for tensorflow lite',
                        metavar='N', default=4, type=int)
    parser.add_argument('--deepsorthome', help='Location of model_data directory',
                        metavar='PATH', default=None)
    parser.add_argument('--camera-flip', help='Flip the camera image vertically',
                        default=False, action='store_true')
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
    parser.add_argument('--mqtt-broker', help='hostname of MQTT broker',
                        default=None, metavar='HOST')
    parser.add_argument('--mqtt-acp-id', help='ACP identity of this MQTT publisher',
                        default=None, metavar='ID')
    parser.add_argument('--mqtt-user', help='username for MQTT login',
                        default=None, metavar='USER')
    parser.add_argument('--mqtt-pass', help='password for MQTT login',
                        default=None, metavar='PASS')
    parser.add_argument('--mqtt-topic', help='topic for MQTT message',
                        default=None, metavar='TOPIC')
    parser.add_argument('--heartbeat-delay-secs', help='seconds between heartbeat MQTT updates',
                        default=60*5, metavar='SECS', type=int)
    parser.add_argument('--disable-background-subtraction', help='Disable background subtraction / motion detection',
                        default=False, action='store_true')
    parser.add_argument('--background-subtraction-ratio', help='Ratio (between 0 and 1) of background motion needed to accept object',
                        default=0.25, metavar='RATIO', type=float)
    parser.add_argument('--enable-background-masking', help='Enable masking of camera view with background subtraction',
                        default=False, action='store_true')
    parser.add_argument('--log', help='Log state of parameters in given file as JSON',
                        default=None, metavar='FILE')
    parser.add_argument('--object-annotation', help='The category of information to show with each detected object (options: ID, LABEL, NONE).',
                        default='LABEL', metavar='CATEGORY', choices=['ID','id','LABEL','label','NONE','none'])
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
    await pipeline.init_mqtt()
    current_app.config.pipeline = pipeline
    cmdserver, protocol = await loop.create_datagram_endpoint(
        lambda: CommandServer(pipeline),
        local_addr=('127.0.0.1', args.control_port))

    # signal handlers
    def shutdown():
        pipeline.running = False
        # When the pipeline finishes, cancel remaining tasks
        for p in asyncio.Task.all_tasks():
            p.cancel()

    loop.add_signal_handler(signal.SIGINT, shutdown)
    loop.add_signal_handler(signal.SIGTERM, shutdown)
    loop.add_signal_handler(signal.SIGHUP, shutdown)

    # Kickstart the main pipeline
    asyncio.ensure_future(pipeline.start())
    asyncio.ensure_future(pipeline.periodic_mqtt_heartbeat())

    # await loop.run_until_complete(asyncio.Task.all_tasks()) # doesn't seem to be needed

@webapp.after_serving
async def shutdown():
    global cmdserver
    cmdserver.close()

if __name__ == '__main__':
    uvloop.install()
    try:
        webapp.run()
    except concurrent.futures.CancelledError:
        pass

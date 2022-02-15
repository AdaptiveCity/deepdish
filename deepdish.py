#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import platform
import re
import io
import psutil
from timeit import time
from time import time, asctime, localtime, sleep
import warnings
import sys
import argparse
import signal
import traceback
from collections import deque

import numpy as np
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
# pylint: disable=g-import-not-at-top
# Be capable of running partial functionality even without all dependencies installed
try:
    from tools.ssd_mobilenet import SSD_MOBILENET
except:
    pass
try:
    from tools.tflite import TFLITE
except:
    pass
try:
    from tools.yolo import YOLO
except:
    pass
try:
    from tools.yolov5 import YOLOV5
except:
    pass
try:
    from tools.saved_model import SAVED_MODEL
except:
    pass
# pylint: enable=g-import-not-at-top

from tools.intersection import any_intersection, intersection
import cameratransform as ct

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from deepdish.framerecords import FrameRecords

import asyncio
import uvloop
import aiofiles
import concurrent.futures
from gmqtt import Client as MQTTClient
import json
import xml.etree.ElementTree as ET

from quart import Quart, Response, current_app
from hypercorn.asyncio import serve
from hypercorn.config import Config
import threading
import faulthandler

faulthandler.enable()

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

def capthread_f(cap, kickstart, box, everyframe, interframe_delay):
    count = 0
    # interframe_delay here is already converted to seconds
    delay = interframe_delay
    try:
        kickstart.wait()
        prev_t = time()
        ret = True
        while ret:
            t1 = time()
            ret, frame = cap.read()
            if not ret:
              frame = None
            t2 = time()
            capthread_delta_t = t2 - prev_t
            #print('{:.02f}ms'.format((t2-prev_t)*1000))
            prev_t = t2
            count += 1
            box.set_message((count,frame,t2,t2-t1))
            # If we are ensuring every frame is processed then wait for
            # synchronising event to be triggered
            if everyframe is not None:
                everyframe.wait()
                everyframe.clear()
            elif interframe_delay is not None and frame is not None:
                # Adjust 'delay' so that the measured capthread_delta_t approaches it
                if capthread_delta_t < interframe_delay:
                    delay+=0.001
                elif capthread_delta_t > interframe_delay:
                    delay-=0.001
                #print('{:.02f}ms {:.02f}ms {:.02f}ms'.format(delay*1000, capthread_delta_t*1000, interframe_delay*1000))
                delay = max(0, delay)
                sleep(delay)
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
    def __init__(self, t_frame, framenum):
        self.t_frame = t_frame
        self.framenum = framenum
        self.priority = 0

    def do_text(self, handle, elements):
        handle.write('Frame {}:'.format(self.framenum))
        for e in elements:
            if isinstance(e, TimingInfo):
                handle.write(' {}={:.0f}ms'.format(e.short_label, e.delta_t*1000))
            elif isinstance(e, TempInfo):
                handle.write(' temp={:.0f}C'.format(e.temp))
            elif isinstance(e, PipelineInfo):
                handle.write(' pipe={} cpup={:.0f}%'.format(e.count, e.cpup))
        handle.write('\n')

    def do_json(self, json):
        json['framenum'] = self.framenum
        json['acp_ts'] = str(self.t_frame)

class TimingInfo:
    def __init__(self, desc, short_label, delta_t):
        self.description = desc
        self.short_label = short_label
        self.delta_t = delta_t
        self.priority = 1

    def do_json(self, json):
        if 'timing' not in json: json['timing'] = {}
        json['timing'][self.short_label]=round(self.delta_t*1000)

class TempInfo:
    def __init__(self, temp):
        self.temp = temp
        self.priority = 2

    def do_json(self, json):
        json['temp']=self.temp

class PipelineInfo:
    def __init__(self, count, qsizes, cpup, freq):
        self.count = count
        self.priority = 3
        self.qsizes = qsizes
        self.cpup = cpup # cpu %
        self.freq = freq # cpu freq

    def do_json(self, json):
        json['pipe']=self.count
        json['qsizes']=self.qsizes
        json['cpup']=self.cpup
        json['freq']=self.freq

# A detected object - simply the information conveyed by the object detector
class DetectedObject:
    def __init__(self, bbox):
        self.bbox = bbox
        self.priority = 5
        self.outline = (255, 0, 0)
    def do_render(self, render):
        pts = list(np.int32(np.array(self.bbox).reshape(-1,2) * render.ratio).reshape(-1))
        render.draw.rectangle(pts, outline=self.outline)
    def do_json(self, json):
        if 'detections' not in json: json['detections'] = []
        json['detections'].append({'bbox': self.bbox.astype(np.int32).tolist()})

# A tracked object based on the output of the tracker
class TrackedObject:
    def __init__(self, bbox, txt, lbl, conf, track_id):
        self.bbox = bbox
        self.txt = txt
        self.label = lbl
        self.track_id = track_id
        self.confidence = conf
        self.priority = 6
        self.outline = (255, 255, 255)
        self.font_fill = (0, 255, 0)
        self.font = 'tiny'
    def do_render(self, render):
        pts = list(np.int32(np.array(self.bbox).reshape(-1,2) * render.ratio).reshape(-1))
        render.draw.rectangle(pts, outline=self.outline)
        render.draw.text(self.bbox[:2],str(self.txt), fill=self.font_fill, font=render.fontlib.fetch(self.font))
    def do_json(self, json):
        if 'tracks' not in json: json['tracks'] = []
        json['tracks'].append({'bbox': self.bbox.astype(np.int32).tolist(), 'label': self.label, 'confidence': self.confidence, 'track_id': self.track_id})


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
        self.raw = True

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

class TopDownView:
    def __init__(self, topdownview):
        (viewpos, viewsize) = topdownview
        self.viewpos = np.array(viewpos,dtype=int)
        self.viewsize = np.array(viewsize,dtype=int)
        self.priority = 9

    def do_render(self, render):
        pts = list(np.array([self.viewpos, self.viewpos + self.viewsize]).reshape(-1))
        render.draw.rectangle(pts, fill=(0, 0, 0))

class TopDownObj():
    def __init__(self, topdownview, pts):
        (viewpos, viewsize) = topdownview
        self.viewpos = np.array(viewpos,dtype=int)
        self.viewsize = np.array(viewsize,dtype=int)
        # transform points into top-down view window
        self.pts = pts.reshape(-1, 2) * np.array([1, -1]) + viewsize * np.array([0.5, 1]) + viewpos
        self.priority = 10
        self.fill = (0, 255, 0)
        self.width = 2

    def do_render(self, render):
        half = np.array([self.width/2.0, self.width/2.0])
        pt0 = self.pts[-1]
        rectpts = list(np.array([pt0 - half, pt0 + half],dtype=int).reshape(-1))
        render.draw.rectangle(rectpts, fill=self.fill)
        linepts = list(np.array(self.pts.reshape(-1),dtype=int))
        render.draw.line(linepts, fill=self.fill, width=self.width)

##################################################

class Pipeline:
    """Object detection and tracking pipeline"""

    def __init__(self, args):
        self.args = args

        # Track the current process
        self.process = psutil.Process()

        # Initialise camera & camera viewport
        self.init_camera()
        # Initialise output
        self.init_output(self.args.output)

        # Process comma-separated list of wanted labels
        self.wanted_labels = self.args.wanted_labels.strip().split(',')

        # Open annotations XML file if it exists
        self.xmltree = None
        if self.args.input_cvat_dir is not None:
            self.annotationfile = os.path.join(self.args.input_cvat_dir, "annotations.xml")
            try:
                self.xmltree = ET.parse(self.annotationfile)
            except FileNotFoundError:
                self.xmltree = None

        self.basedir = self.args.basedir
        model = os.path.join(self.basedir, self.args.model)
        if self.args.labels:
            labels = os.path.join(self.basedir, self.args.labels)
        else:
            labels = None
        # Initialise object detector (for some reason it has to happen
        # here & not within detect_objects(), or else the inference engine
        # gets upset and starts throwing NaNs at me. Thanks, Python.)

        use_edgetpu = 'edgetpu' in self.args.model and not self.args.disable_edgetpu

        if 'yolov5' in self.args.model:
            self.object_detector = YOLOV5(wanted_labels=self.wanted_labels, model_file=model, label_file=labels, num_threads=self.args.num_threads, edgetpu=use_edgetpu)
        elif 'yolo' in self.args.model:
            self.object_detector = YOLO(wanted_labels=self.wanted_labels, model_file=model, label_file=labels, num_threads=self.args.num_threads)
        elif 'saved_model' in self.args.model:
            self.object_detector = SAVED_MODEL(wanted_labels=self.wanted_labels, model_file=model, label_file=labels, num_threads=self.args.num_threads)
        elif 'mobilenet' in self.args.model:
            self.object_detector = SSD_MOBILENET(wanted_labels=self.wanted_labels, model_file=model, label_file=labels,
                    num_threads=self.args.num_threads, edgetpu=use_edgetpu)
        elif 'tflite' in self.args.model:
            self.object_detector = TFLITE(wanted_labels=self.wanted_labels, model_file=model, label_file=labels,
                    num_threads=self.args.num_threads, edgetpu=use_edgetpu)
        elif use_edgetpu:
            from tools.edgetpu import EDGETPU
            self.object_detector = EDGETPU(wanted_labels=self.wanted_labels, model_file=model, label_file=labels,
                    num_threads=self.args.num_threads, edgetpu=use_edgetpu)
        else:
            print('Unsure what to do with model file {}'.format(self.args.model))
            sys.exit(1)

        # Initialise feature encoder
        if self.args.encoder_model is None:
            model_filename = os.path.join(self.args.deepsorthome, 'mars-64x32x3.pb')
        else:
            model_filename = os.path.join(self.args.deepsorthome, self.args.encoder_model)

        self.encoder = gdet.create_box_encoder(model_filename, batch_size=self.args.encoder_batch_size, num_threads=self.args.num_threads)

        self.background_subtraction = not self.args.disable_background_subtraction

        # Initialise tracker
        nn_budget = None
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", self.args.max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric,max_iou_distance=self.args.max_iou_distance, max_age=self.args.max_age)

        # Initialise database
        self.db = {}
        self.data_lock = asyncio.Lock()
        self.framenum_committed = 0 # The frame number associated with
                                    # the information kept in the
                                    # following variables
        self.delcount = dict([(lbl, 0) for lbl in self.wanted_labels])
        self.intcount = dict([(lbl, 0) for lbl in self.wanted_labels])
        self.poscount = dict([(lbl, 0) for lbl in self.wanted_labels])
        self.negcount = dict([(lbl, 0) for lbl in self.wanted_labels])

        self.mqtt = None
        self.topic = self.args.mqtt_topic
        self.mqtt_acp_id = self.args.mqtt_acp_id
        self.heartbeat_delay_secs = self.args.heartbeat_delay_secs

        self.pipeline_sem = asyncio.Semaphore()
        self.final_frame = None # Not set until the final frame is
                                # reached and known, if ever.

        self.frame_count = 0 # self.frame_count is only used to assign
                             # the next 'framenum' number. It should
                             # not be read otherwise because
                             # pipelining means there could be
                             # overlapping stages & race conditions.

        self.log = self.args.log
        if self.log is not None:
            if self.args.restore_from_log and os.path.exists(self.log):
                with open(self.log, mode='r') as f:
                    q = deque(f, 1)
                    if len(q) > 0:
                        last_line = q.pop()
                        data = json.loads(last_line)
                        for lbl in self.wanted_labels:
                            self.poscount[lbl] = data.get('poscount_'+lbl, 0)
                            self.negcount[lbl] = data.get('negcount_'+lbl, 0)
                            self.delcount[lbl] = data.get('delcount_'+lbl, 0)
                            self.intcount[lbl] = data.get('intcount_'+lbl, 0)
                        self.frame_count = data.get('frame_count', 0)
            else:
                with open(self.log, mode='w+') as f:
                    f.truncate()
        self.loop = asyncio.get_event_loop()
        self.t_prev = None # frame to frame times

        self.cpu_temp_file = '/sys/class/thermal/thermal_zone0/temp'
        if self.args.cpu_temp_file is not None:
            self.cpu_temp_file = self.args.cpu_temp_file
        if not os.path.exists(self.cpu_temp_file):
            self.cpu_temp_file = None

        self.cpu_freq_file = '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'
        if self.args.cpu_freq_file is not None:
            self.cpu_freq_file = self.args.cpu_freq_file
        if not os.path.exists(self.cpu_freq_file):
            self.cpu_freq_file = None
            self.cpu_gov_file = None
        else:
            self.cpu_gov_file = os.path.join(os.path.dirname(self.cpu_freq_file), 'scaling_governor')
            if not os.path.exists(self.cpu_gov_file):
                self.cpu_gov_file = None

        self.powersave_delay = 0
        self.powersave_delay_maximum = float(self.args.powersave_delay_maximum) / 1000.0
        if self.args.disable_powersaving:
            self.powersave_delay_increment = 0
        else:
            self.powersave_delay_increment = float(self.args.powersave_delay_increment) / 1000.0

        self.cam = None
        self.topdownview = None
        self.topdownview_scalefactors = None
        if self.args.three_d:
            if self.args.focallength_mm is not None and self.args.sensor_width_mm is not None and self.args.sensor_height_mm is not None and self.args.elevation_m is not None and self.args.tilt_deg is not None:
                (w, h) = self.input_size
                self.cam = ct.Camera(ct.RectilinearProjection(focallength_mm=self.args.focallength_mm,
                                                              sensor=(self.args.sensor_width_mm, self.args.sensor_height_mm),
                                                              image=self.input_size),
                                     ct.SpatialOrientation(elevation_m=self.args.elevation_m,
                                                           tilt_deg=self.args.tilt_deg,
                                                           roll_deg=self.args.roll_deg))
                defaultviewsize = ((0, 0), (w/4, h/4))
                if self.args.topdownview_size_m is not None:
                    size = np.array(list(map(int,self.args.topdownview_size_m.strip().split(','))),dtype=float)
                    scalefactors = np.array(defaultviewsize[1],dtype=float) / size
                    self.topdownview = defaultviewsize
                    self.topdownview_scalefactors = scalefactors
                else:
                    self.topdownview = defaultviewsize
                    self.topdownview_scalefactors = np.array([1,1])
            else:
                raise Error('3-D transform requires focallength, sensor size, camera elevation and tilt.')

        # Initialise frame recording system
        self.framerec = FrameRecords(self.object_detector.labels)

        # Examine CVAT-format XML file if given
        if self.xmltree is not None:
            # Compare labels in XML spec to labels in labelfile
            full_labels = self.object_detector.labels
            labels_to_id = {v: k for k, v in full_labels.items()}

            for l in self.xmltree.getroot().findall('./meta/task/labels/label'):
                name = l.find('name').text
                id = labels_to_id.get(name, None)
                color = l.find('color').text
                # print("Annotation label '{}' mapped to detector label '{}' ID '{}', assigned color '{}'".format(name, full_labels.get(id, None), id, color))
                # print("To change this try argument: --remap-annotation-labels 'annotation label:detector label,...'")
                self.framerec.add_annotation_label_info(name, id, color)
            for t in self.xmltree.getroot().findall('./track'):
                lblname = t.get('label')
                track_id = int(t.get('id'))
                # print("Track {} label='{}'".format(track_id, lblname))
                for b in t.findall('box'):
                    frame=int(b.get('frame'))
                    outside=b.get('outside')=='1'
                    occluded=b.get('occluded')=='1'
                    keyframe=b.get('keyframe')=='1'
                    pts = np.array([b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')], dtype=float)
                    z_order = int(b.get('z_order'))
                    # print("box frame={} outside={:d} occluded={:d} keyframe={:d} pts={} z_order={}".format(frame,outside,occluded,keyframe,pts,z_order))
                    self.framerec.add_annotated_track(frame, track_id, lblname, pts, outside, occluded, keyframe, z_order)

    def on_mqtt_connect(self, client, flags, rc, properties):
        self.mqtt_connect_event.set()
        if self.args.mqtt_verbosity > 1:
            payload = {'acp_ts': str(time()), 'acp_event': 'initialisation', 'acp_id': self.mqtt_acp_id,
                       'model': self.args.model, 'model_class': type(self.object_detector).__name__,
                       'encoder_model': self.args.encoder_model, 'encoder_model_class': type(self.encoder.image_encoder).__name__,
                       'input': self.input,
                       'use_edgetpu': self.object_detector.use_edgetpu,
                       'input_shape': [self.object_detector.width, self.object_detector.height],
                       'encoder_input_shape': [self.encoder.width, self.encoder.height],
                       'num_threads': self.object_detector.num_threads,
                       'max_age': self.args.max_age,
                       'max_iou_distance': self.args.max_iou_distance,
                       'nms_max_overlap': self.args.nms_max_overlap,
                       'max_cosine_distance': self.args.max_cosine_distance,
                       'background_subtraction': None if self.args.disable_background_subtraction else self.args.background_subtraction_ratio,
                       'powersaving': None if self.args.disable_powersaving else (self.args.powersave_delay_increment, self.args.powersave_delay_maximum),
                       'cpu_governor': self.cpu_governor,
                       'object_detector_skip_frames': self.args.object_detector_skip_frames,
                       'interframe_delay': self.args.interframe_delay
                       }
            self.mqtt.publish(self.topic, json.dumps(payload))

    async def init_mqtt(self):
        self.cpu_governor = await self.get_cpu_governor() # only used in MQTT messages anyway
        if self.args.mqtt_broker is not None:
            self.mqtt = MQTTClient('deepdish-'+platform.node())
            if self.topic is None:
                self.topic = 'default/topic'
            self.mqtt_connect_event = asyncio.Event()
            self.mqtt.on_connect = self.on_mqtt_connect
            self.mqtt.set_config({'reconnect_retries': 10, 'reconnect_delay': 1})
            if self.args.mqtt_user is not None:
                self.mqtt.set_auth_credentials(self.args.mqtt_user, self.args.mqtt_pass)
            print('Waiting to connect to MQTT broker.')
            await self.mqtt.connect(self.args.mqtt_broker, self.args.mqtt_port)
            await self.mqtt_connect_event.wait()

    def init_camera(self):
        self.input = self.args.input
        if self.args.input_cvat_dir is not None:
            # Set up frame-by-frame from files in input CVAT directory
            self.input = os.path.join(self.args.input_cvat_dir, "images/frame_%06d.jpg")
            # Open test file
            with Image.open(self.input % 1) as im:
                self.input_size = im.size
            # Capture every frame from the video file / dir
            self.everyframe = threading.Event()
            # Disable power-saving delay mechanism
            self.args.disable_powersaving = True
            self.powersave_delay_increment = 0
        elif self.input is None:
            self.input_size = (self.args.camera_width, self.args.camera_height)
            if self.args.gstreamer is not None:
                src = self.args.gstreamer
            elif self.args.gstreamer_nvidia:
                src = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int){}, height=(int){}, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true".format(*self.input_size)
            else:
                src = self.args.camera
            self.input = src
            # Allow live camera frames to be dropped
            self.everyframe = None
        else:
            # Capture every frame from the video file in self.input
            if self.args.interframe_delay is None:
                self.everyframe = threading.Event()
            # Disable power-saving delay mechanism
            self.args.disable_powersaving = True
            self.powersave_delay_increment = 0

        # Set up the OpenCV video capture
        self.cap = cv2.VideoCapture(self.input)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Just in case input_size wasn't already set up
        self.input_size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        # Configure the 'counting line' in the camera viewport
        if self.args.line is None:
            w, h = self.input_size
            self.countline = np.array([[w/2,0],[w/2,h]],dtype=int)
        else:
            self.countline = np.array(list(map(int,self.args.line.strip().split(','))),dtype=int).reshape(2,2)
        self.cameracountline = self.countline.astype(float)

    def init_output(self, output):
        if self.args.disable_graphics:
            self.output = None
            return
        self.color_mode = None # fixme
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            fps = 15 # FIXME: no way of determining FPS
        w, h = self.input_size

        self.backbuf = Image.new("RGBA", self.input_size, (0,0,0,0))
        self.draw = ImageDraw.Draw(self.backbuf)
        if self.args.output_cvat_dir is None:
            self.output = cv2.VideoWriter(self.args.output,fourcc, fps, self.input_size)
        else:
            # write individual frame files in CVAT format
            outpath = os.path.join(self.args.output_cvat_dir,'images','frame_%06d.jpg')
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            self.output = cv2.VideoWriter(outpath, 0, 0, self.input_size)
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
            # Framebuffer size can be different than input/output size
            (w, h) = (self.args.framebuffer_width, self.args.framebuffer_height)
            if w is None or h is None:
                nums = re.findall('(.*),(.*)', open(vsizefile).read())[0]
                if w is None:
                    w = int(nums[0])
                if h is None:
                    h = int(nums[1])
            self.framebufres = (w, h)
            print('Framebuffer device: {} resolution: {},{}'.format(self.framebufdev,w,h))

    def shutdown(self):
        global shutdown_event
        self.running = False
        print('Shutting down pipeline.')
        if self.args.output_cvat_dir is not None:
            print('Writing CVAT output.')
            # Write CVAT-format annotations XML file if possible
            if self.xmltree is not None:
                meta = self.xmltree.getroot().find('./meta')
            else:
                meta = None
            xmlout = self.framerec.xml_output(meta=meta)
            xmloutfile = os.path.join(self.args.output_cvat_dir, "annotations.xml")
            with open(xmloutfile, mode='wb') as f:
                xmlout.write(f, xml_declaration=True, encoding='utf-8', short_empty_elements=False)
        if cmdserver:
            print('Shutting down command server.')
            cmdserver.close()
        if self.mqtt:
            print('Shutting down MQTT client.')
            if self.args.mqtt_verbosity > 1:
                payload = {'acp_ts': str(time()), 'acp_event': 'shutdown', 'acp_id': self.mqtt_acp_id, 'model': self.args.model, 'input': self.input}
                self.mqtt.publish(self.topic, json.dumps(payload))
        print('Shutting down Quart server.')
        shutdown_event.set()

    async def get_cpu_temp(self):
        if not self.cpu_temp_file: return None
        async with aiofiles.open(self.cpu_temp_file, mode='r') as f:
            line = await f.read()
            temp = float(line)
            return temp/1000

    async def get_cpu_freq(self):
        if not self.cpu_freq_file: return None
        async with aiofiles.open(self.cpu_freq_file, mode='r') as f:
            line = await f.read()
            freq = int(line)
            return freq

    async def get_cpu_governor(self):
        if not self.cpu_gov_file: return None
        async with aiofiles.open(self.cpu_gov_file, mode='r') as f:
            gov = await f.read()
            return gov.strip()

    async def capture(self, q, box):
        try:
            # The purpose of this loop is to decouple the capthread from the pipeline.

            # When dealing with a live video stream (having
            # everyframe=False) then we must pull frames off the
            # live camera as fast as they appear, or else OpenCV
            # starts queueing them up internally (a rather poor
            # design) and we fall behind 'real time'.
            while self.running:
                # Fetch frame from box where capthread has placed it
                frame = None
                msg = None
                while msg is None:
                    await asyncio.sleep(0.003) # cooperative yield
                    # note that .sleep(0) doesn't work right, causing severely inconsistent timings
                    msg = box.get_message()
                (orig_framenum, frame, t_frame, dt_cap) = msg

                if self.everyframe:
                    box.set_message(None) # avoid repeating frames

                if frame is None:
                    self.final_frame = self.frame_count
                    break

                if self.args.camera_flip:
                    # If we need to flip the image vertically
                    frame = cv2.flip(frame, 0)
                # Ensure frame is proper size
                frame = cv2.resize(frame, self.input_size)

                # q is a 1-element FreshQueue that overwrites the existing element if there is one
                q.put_nowait((orig_framenum, frame, dt_cap, t_frame, time()))

                # slow down pipeline if trying to save power
                if self.powersave_delay > 0:
                    await asyncio.sleep(self.powersave_delay)

        finally:
            if self.cap is not None:
                self.cap.release()

    def run_object_detector(self, frame):
        t1 = time()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))
        (boxes, labels, scores) = self.object_detector.detect_image(image)
        t2 = time()
        return (boxes, labels, scores, t2 - t1)

    async def detect_objects(self, q_in, q_out):
        # Initialise background subtractor
        backSub = cv2.createBackgroundSubtractorMOG2()

        # Initialise vars for skipping a set number of frames between objd invocations:
        skip_rem = 0
        prev_objd_result = None

        # Feed some dummy data to warm-up the object detector and feature encoder
        dummyframe = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        await self.loop.run_in_executor(None, self.run_object_detector, dummyframe)
        await self.loop.run_in_executor(None, self.encoder, dummyframe, [(0, 0, self.encoder.width, self.encoder.height)])

        # Now we're ready to start the capthread:
        self.kickstart.set()

        while self.running:
            # Obtain next video frame
            (orig_framenum, frame, dt_cap, t_frame, t_prev) = await q_in.get()
            if orig_framenum <= self.frame_count:
                # We've already seen this frame
                await asyncio.sleep(0.003) # cooperative yield
                continue
            t_frame_recv = time()
            framenum = orig_framenum
            self.frame_count = orig_framenum
            # Frame num. 'framenum' begins its journey through the pipeline here
            self.pipeline_sem.release()

            if self.everyframe:
                # Notify other side that this frame is in the pipeline
                self.everyframe.set()

            # Apply background subtraction to find image-mask of areas of motion
            if self.background_subtraction:
                fgMask = backSub.apply(frame)
                if self.args.enable_background_masking:
                    frame = cv2.bitwise_and(frame,frame,mask = fgMask)
            # Convert to PIL Image
            #image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))
            t_backsub = time()

            if skip_rem > 0 and prev_objd_result is not None:
                (boxes0, labels0, scores0, delta_t) = prev_objd_result
                skip_rem-=1
                objd_skipped = True
            else:
                # Run object detection engine within a Thread Pool
                (boxes0, labels0, scores0, delta_t) = await self.loop.run_in_executor(None, self.run_object_detector, frame)
                prev_objd_result = (boxes0, labels0, scores0, delta_t)
                skip_rem = self.args.object_detector_skip_frames or 0
                objd_skipped = False

            # Filter object detection boxes, including only those with areas of motion
            t1 = time()
            boxes = []
            labels = []
            scores = []
            max_x, max_y = self.input_size
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

            # start slowing down the pipeline if there are no objects in scene
            if not self.args.disable_powersaving and len(boxes) == 0:
                self.powersave_delay += self.powersave_delay_increment
                if self.powersave_delay > self.powersave_delay_maximum:
                    self.powersave_delay = self.powersave_delay_maximum
            else:
                self.powersave_delay = 0

            # Send results to next step in pipeline
            elements = [FrameInfo(t_frame, framenum),
                        CameraImage(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGRA2RGB), mode='RGB')),
                        CameraCountLine(self.cameracountline),
                        TimingInfo('Frame capture latency', 'fcap', dt_cap),
                        TimingInfo('Frame return [Q0] latency', 'fram', t_prev - t_frame),
                        TimingInfo('Frame / Q1 item received latency', 'q1', t_frame_recv - t_prev),
                        #TimingInfo('Frame prep latency', 'prep', t_prep - t_frame_recv),
                        TimingInfo('Background subtraction latency', 'bsub', t_backsub - t_frame_recv)]
            if not objd_skipped:
                elements.append(TimingInfo('Object detection latency', 'objd', delta_t+(t2-t1)))
            await q_out.put((frame, framenum, boxes, labels, scores, objd_skipped, elements, time()))

    async def encode_features(self, q_in, q_out):
        with concurrent.futures.ThreadPoolExecutor() as pool:
            prev_features = None
            while self.running:
                # Obtain next video frame and object detection boxes
                (frame, framenum, boxes, labels, scores, objd_skipped, elements, t_prev) = await q_in.get()

                t1 = time()
                # Run non-max suppression to eliminate spurious boxes
                boxesA0 = np.array(boxes)
                scoresA0 = np.array(scores)
                indices = preprocessing.non_max_suppression(boxesA0, self.args.nms_max_overlap, scoresA0)
                boxesA1 = boxesA0[indices]
                scoresA1 = scoresA0[indices]
                labels1 = [labels[i] for i in indices]

                # Consider and modify boxes based on info contained in the frame record
                boxesA2, labels2, scoresA2 = self.framerec.process_boxes(framenum, boxesA1, labels1, scoresA1)

                if objd_skipped and prev_features is not None:
                    features = prev_features
                    feat_skipped = True
                else:
                    # Run feature encoder within a Thread Pool
                    features, ienc_dt = await self.loop.run_in_executor(pool, self.encoder, frame, boxesA2, True)
                    prev_features = features
                    feat_skipped = False
                t2 = time()

                # Build list of 'Detection' objects and send them to next step in pipeline
                detections = [Detection(bbox, lbl, scr, feature) for bbox, lbl, scr, feature in zip(boxesA2, labels2, scoresA2, features)]

                # Consider and modify detections based on info contained in the frame record
                detections = self.framerec.process_detections(framenum, detections)
                elements.append(TimingInfo('Q1 / Q2 latency', 'q2', (t1 - t_prev)))
                if not feat_skipped:
                    elements.append(TimingInfo('Internal encoder latency', 'ienc', ienc_dt))
                    elements.append(TimingInfo('Feature encoder latency', 'feat', (t2-t1)))
                await q_out.put((framenum, detections, elements, time()))

    async def track_objects(self, q_in, q_out):
        while self.running:
            (framenum, detections, elements, t_prev) = await q_in.get()
            t1 = time()
            self.tracker.predict()
            self.tracker.update(detections)
            t2 = time()
            elements.append(TimingInfo('Q2 / Q3 latency', 'q3', (t1 - t_prev)))
            elements.append(TimingInfo('Tracker latency', 'trak', (t2-t1)))
            await q_out.put((framenum, detections, elements, time()))

    async def process_results(self, q_in, q_out):
        while self.running:
            (framenum, detections, elements, t_prev) = await(q_in.get())

            t1=time()
            delcounts={}
            for track in self.tracker.deleted_tracks:
                i = track.track_id
                if track.is_deleted():
                    delcounts = self.check_deleted_track(track)

            # Consider and modify tracks based on info in frame record
            self.tracker.tracks = self.framerec.process_tracking(framenum, self.tracker)

            intersects = [] # accumulate list of intersection events
            for track in self.tracker.tracks:
                i = track.track_id
                lbl, conf = track.get_label(return_confidence=True)
                if not track.is_confirmed() or track.time_since_update > 1:
                    # track was not updated this frame, or it was not confirmed
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
                        # accumulate intersection events for later processing
                        intersects.append({'label': lbl, 'element': TrackedPathIntersection(pts[-4:]), 'cp': cp})

                if self.args.object_annotation.lower() == 'id':
                    annot = str(track.track_id)
                elif self.args.object_annotation.lower() == 'label':
                    annot = lbl
                else:
                    annot = ''
                elements.append(TrackedObject(bbox, annot, lbl, conf, track.track_id))

                if self.cam is not None and self.topdownview is not None:
                    # Add to top-down view using cameratransform
                    pt = self.cam.spaceFromImage(bottomCentre)[:2]
                    pts_pretransform = self.cam.spaceFromImage(np.array(self.db[i]))
                    if self.topdownview_scalefactors is not None:
                        pts_postransform = self.topdownview_scalefactors * pts_pretransform[:,:2]
                        pts = pts_postransform[:,:2].reshape(-1)
                    else:
                        pts = pts_pretransform[:,:2].reshape(-1)
                    elements.append(TopDownObj(self.topdownview,pts))

            async with self.data_lock:
                # update global state within lock
                for inter in intersects:
                    lbl = inter['label']
                    if inter['cp'] >= 0:  # check cross product for direction
                        self.poscount[lbl] += 1
                    else:
                        self.negcount[lbl] += 1
                    self.intcount[lbl] += 1
                    print("track_id={} ({}) just intersected camera countline; cross-prod={}; intcount={}".format(i,lbl,cp,self.intcount))

                for lbl, delta in delcounts.items():
                    self.delcount[lbl] += delta
                    print("delcount[{}]={}".format(lbl,self.delcount[lbl]))

                self.framenum_committed = framenum

            # fire off I/O-related events that occur after intersections are detected
            for inter in intersects:
                if inter['cp'] >= 0:
                    crossing_type = 'pos'
                else:
                    crossing_type = 'neg'
                elements.append(inter['element'])
                await self.publish_crossing_event(elements, crossing_type)

            for det in detections:
                bbox = det.to_tlbr()
                elements.append(DetectedObject(bbox))


            if self.topdownview is not None:
                # Draw background for top-down view
                elements.append(TopDownView(self.topdownview))

            elements.append(CountingStats(self.negcount, self.poscount))
            t2=time()
            elements.append(TimingInfo('Q3 / Q4 latency', 'q4', (t1-t_prev)))
            elements.append(TimingInfo('Results processing latency', 'proc', (t2-t1)))

            await q_out.put((elements,time()))

    def update_payload_with_state(self, payload):
        for lbl in self.wanted_labels:
            payload.update(dict([
                ('poscount_'+lbl, self.poscount[lbl]), ('negcount_'+lbl, self.negcount[lbl]), ('diff_'+lbl, self.poscount[lbl] - self.negcount[lbl]),
                ('intcount_'+lbl, self.intcount[lbl]), ('delcount_'+lbl, self.delcount[lbl]) ]))

    async def publish_crossing_event(self, elements, crossing_type):
        for e in elements:
            if isinstance(e, FrameInfo):
                t_frame = e.t_frame
                count = e.framenum
                break

        temp = await self.get_cpu_temp()
        if self.mqtt is not None and self.args.mqtt_verbosity > 0:
            payload = {'acp_ts': str(t_frame), 'acp_id': self.mqtt_acp_id, 'acp_event': 'crossing', 'acp_event_value': crossing_type, 'temp': temp}
            async with self.data_lock:
                self.update_payload_with_state(payload)
            self.mqtt.publish(self.topic, json.dumps(payload))

        if self.log is not None:
            payload = {'timestamp': str(t_frame), 'asctime': asctime(localtime(t_frame)), 'frame_count': count, 'temp': temp}
            async with self.data_lock:
                self.update_payload_with_state(payload)
            async with aiofiles.open(self.log, mode='a+') as f:
                await f.write(json.dumps(payload) + '\n')

    async def periodic_heartbeat(self):
        while True:
            temp = await self.get_cpu_temp()
            if self.mqtt is not None and self.args.mqtt_verbosity > 0:
                payload = {'acp_ts': str(time()), 'acp_id': self.mqtt_acp_id, 'acp_event': 'heartbeat', 'temp': temp}
                async with self.data_lock:
                    self.update_payload_with_state(payload)
                self.mqtt.publish(self.topic, json.dumps(payload))

            if self.log is not None:
                payload = {'timestamp': str(time()), 'asctime': asctime(), 'temp': temp}
                async with self.data_lock:
                    payload['frame_count'] = self.framenum_committed
                    self.update_payload_with_state(payload)
                async with aiofiles.open(self.log, mode='a+') as f:
                    await f.write(json.dumps(payload) + '\n')

            await asyncio.sleep(self.heartbeat_delay_secs)

    async def graphical_output(self, render : RenderInfo, elements, output_wh : (int, int)):
        (output_w, output_h) = output_wh

        # Clear screen
        self.draw.rectangle([0, 0, output_w, output_h], fill=0, outline=0)

        # Sort elements by display priority
        elements.sort(key=lambda e: e.priority)

        # Draw elements
        for e in elements:
            if hasattr(e, 'do_render') and (not self.args.raw_output or getattr(e, 'raw', False)):
                e.do_render(render)

        # Copy backbuf to output
        backarray = np.array(self.backbuf)
        if self.color_mode is not None:
            outputbgra = cv2.cvtColor(backarray, self.color_mode)
        else:
            outputbgra = cv2.cvtColor(backarray, cv2.COLOR_RGBA2BGRA)
        outputbgr = cv2.cvtColor(outputbgra, cv2.COLOR_BGRA2BGR)
        if self.output is not None:
            self.output.write(outputbgr)
        if self.framebufdev is not None:
            outputrgba = cv2.cvtColor(outputbgra, cv2.COLOR_BGRA2RGBA)
            output565 = cv2.cvtColor(outputbgra, cv2.COLOR_RGBA2BGR565)
            outputfbuf = cv2.resize(output565, self.framebufres)
            try:
                with open(self.framebufdev, 'wb') as buf:
                    buf.write(outputfbuf)
            except Exception as e:
                print(type(e))
                print(e.args)
                print(e)
                print('failed to write to framebuffer device {} ...disabling it.'.format(self.framebufdev))
                self.framebufdev = None
        await streaminfo.set_frame(outputbgra)

    def text_output(self, handle, elements):
        # Sort elements by priority
        elements.sort(key=lambda e: e.priority)

        for e in elements:
            if hasattr(e, 'do_text'):
                e.do_text(handle, elements)

        # per-frame MQTT messages if enabled
        if self.mqtt and self.args.mqtt_verbosity > 1:
            payload = {'acp_event': 'frame', 'acp_id': self.mqtt_acp_id}
            for e in elements:
                if hasattr(e, 'do_json'):
                    e.do_json(payload)
            self.mqtt.publish(self.topic, json.dumps(payload))

    async def render_output(self, q_in):
        (output_w, output_h) = self.input_size
        ratio = 1 #fixme
        if not self.args.disable_graphics:
            render = RenderInfo(ratio, FontLib(output_w), self.draw, self.backbuf)

        try:
            while self.running:
                try:
                    (elements, t_prev) = await asyncio.wait_for(q_in.get(), 1)
                except asyncio.TimeoutError:
                    # workaround the race condition wherein
                    # self.final_frame does not get set (in
                    # capthread_f) before q_in.get() is called above
                    if self.final_frame:
                        break
                    continue

                t1 = time()
                if not self.args.disable_graphics:
                    await self.graphical_output(render, elements, (output_w, output_h))

                framenum = None
                for e in elements:
                    if isinstance(e, FrameInfo):
                        framenum = e.framenum
                        t_frame = e.t_frame
                        break
                elements.append(TimingInfo('Q4 / Q5 latency', 'q5', t1 - t_prev))
                elements.append(TimingInfo('Graphical display latency', 'disp', time() - t1))
                t_sum = 0
                for e in elements:
                    if isinstance(e, TimingInfo):
                        t_sum += e.delta_t
                elements.append(TimingInfo('Latency sum', 'sum', t_sum))
                t_now = time()
                t_e2e = t_now - t_frame
                elements.append(TimingInfo('End to end latency', 'e2e', t_e2e))
                elements.append(TimingInfo('Missing', 'miss', t_e2e - t_sum))
                if self.t_prev is not None:
                  elements.append(TimingInfo('Frame to frame latency', 'f2f', t_now - self.t_prev))
                self.t_prev = t_now

                temp = await self.get_cpu_temp()
                elements.append(TempInfo(temp))

                await (self.pipeline_sem.acquire())
                frames_in_flight = self.pipeline_sem._value
                cpup = self.process.cpu_percent()
                freq = await self.get_cpu_freq()
                elements.append(PipelineInfo(frames_in_flight, [q.qsize() for q in self.queues], cpup, freq))

                self.text_output(sys.stdout, elements)

                # Check if we're done with all frames
                if self.final_frame:
                    if framenum == self.final_frame:
                        break

        finally:
            if self.output:
                self.output.release()

    def check_deleted_track(self, track):
        i = track.track_id
        delcounts = {}
        if i in self.db and len(self.db[i]) > 1:
            if any_intersection(self.cameracountline[0], self.cameracountline[1], np.array(self.db[i])):
                l = track.get_label()
                if l not in delcounts: delcounts[l] = 0
                delcounts[l]+=1
            self.db[i] = []
        return delcounts

    async def start(self):
        self.running = True
        cameraQueue = FreshQueue()
        k = self.args.max_queue_size
        objectQueue = asyncio.Queue(maxsize=k)
        detectionQueue = asyncio.Queue(maxsize=k)
        resultQueue = asyncio.Queue(maxsize=k)
        drawQueue = asyncio.Queue(maxsize=k)
        self.queues = [cameraQueue, objectQueue, detectionQueue, resultQueue, drawQueue]

        render_task = asyncio.ensure_future(self.render_output(drawQueue))
        asyncio.ensure_future(self.process_results(resultQueue, drawQueue))
        asyncio.ensure_future(self.track_objects(detectionQueue, resultQueue))
        asyncio.ensure_future(self.encode_features(objectQueue, detectionQueue))
        asyncio.ensure_future(self.detect_objects(cameraQueue, objectQueue))

        # Box that holds frame data and info for communication to main process
        box = MBox()
        # Event that kicks off the capture loop (only when pipeline is ready)
        self.kickstart = threading.Event()
        ifd = self.args.interframe_delay
        if ifd is not None:
            self.everyframe = None
            ifd_sec = float(ifd)/1000.0
        else:
            ifd_sec = None
        capthread = threading.Thread(target=capthread_f, args=(self.cap,self.kickstart,box,self.everyframe,ifd_sec), daemon=True)
        capthread.start()
        self.process.cpu_percent() # take first 'dummy reading' to start monitoring
        await self.capture(cameraQueue, box)
        await render_task
        self.shutdown()

def quoted_split(s):
    def strip_quotes(s):
        if s and (s[0] == '"' or s[0] == "'") and s[0] == s[-1]:
            return s[1:-1]
        return s
    return [strip_quotes(p).replace('\\"', '"').replace("\\'", "'") \
            for p in re.findall(r'(?:[^"\s]*"(?:\\.|[^"])*"[^"\s]*)+|(?:[^\'\s]*\'(?:\\.|[^\'])*\'[^\'\s]*)+|[^\s]+', s)]

def get_arguments():
    basedir = os.getenv('DEEPDISHHOME','.')
    optfileparser = argparse.ArgumentParser()
    optfileparser.add_argument('--options-file', help="Read some command-line options from file also.",
                               metavar='FILE', action='append', default=None)
    argv = sys.argv[1:]
    optfiles = []
    while True:
        (optfileargs, argv2) = optfileparser.parse_known_args(args=argv)
        if optfileargs.options_file:
            argv1 = []
            for n in optfileargs.options_file:
                if n not in optfiles: # prevent infinite loops of includes
                    optfiles.append(n)
                    with open(n) as f:
                        lines = [quoted_split(l) for l in f.readlines() if l.strip()[0] != '#']
                    for l in lines:
                        argv1.extend(l)
            argv1.extend(argv2)
            argv = argv1
        else:
            argv = argv2
            break

    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', help="camera number for live input (OpenCV numbering)",
                        metavar='N', default=0, type=int)
    parser.add_argument('--gstreamer', help='gstreamer pipeline for camera input (instead of camera number)',
                        metavar='PIPELINE', default=None)
    parser.add_argument('--gstreamer-nvidia', help='use nvidia-default gstreamer pipeline (instead of camera number)',
                        action='store_true', default=False)
    parser.add_argument('--input', help="input MP4 file for video file input (instead of camera)",
                        default=None)
    parser.add_argument('--input-cvat-dir', help="input CVAT-format data directory (instead of camera)",
                        default=None)
    parser.add_argument('--output', help="output file with annotated video frames",
                        default=None)
    parser.add_argument('--output-cvat-dir', help="output annotations to CVAT-format data directory",
                        default=None)
    parser.add_argument('--line', '-L', help="counting line: x1,y1,x2,y2",
                        default=None)
    parser.add_argument('--model', help='Path to object detection file or directory.', metavar='FILE', required=True)
    parser.add_argument('--disable-edgetpu', help='Disable any usage of Edge TPU accelerator altogether.',
                        default=False, action='store_true')
    parser.add_argument('--encoder-model', help='Path to feature encoder file.', metavar='FILE')
    parser.add_argument('--encoder-batch-size', help='Batch size for feature encoder inference',
                        default=32, type=int, metavar='N')
    parser.add_argument('--labels', help='Path to labels file.', metavar='FILE', default=None)
    parser.add_argument('--framebuffer', help='Enable framebuffer display', default=False, action='store_true')
    parser.add_argument('--framebuffer-device', '-F', help='Framebuffer device',
                        default='/dev/fb0', metavar='DEVICE')
    parser.add_argument('--framebuffer-width', help='Framebuffer device resolution (width) override',
                        default=None, metavar='WIDTH',type=int)
    parser.add_argument('--framebuffer-height', help='Framebuffer device resolution (height) override',
                        default=None, metavar='HEIGHT',type=int)
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
    parser.add_argument('--disable-graphics', help='Disable all graphical output',
                        default=False, action='store_true')
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
    parser.add_argument('--mqtt-port', help='port of MQTT broker',
                        default=1883, metavar='PORT')
    parser.add_argument('--mqtt-acp-id', help='ACP identity of this MQTT publisher',
                        default=None, metavar='ID')
    parser.add_argument('--mqtt-user', help='username for MQTT login',
                        default=None, metavar='USER')
    parser.add_argument('--mqtt-pass', help='password for MQTT login',
                        default=None, metavar='PASS')
    parser.add_argument('--mqtt-topic', help='topic for MQTT message output',
                        default=None, metavar='TOPIC')
    parser.add_argument('--mqtt-verbosity', help='0=quiet; 1=intersection events; 2=everything',
                        default=1, type=int, metavar='LEVEL')
    parser.add_argument('--heartbeat-delay-secs', help='seconds between heartbeat MQTT updates',
                        default=60*5, metavar='SECS', type=int)
    parser.add_argument('--disable-background-subtraction', help='Disable background subtraction / motion detection',
                        default=False, action='store_true')
    parser.add_argument('--background-subtraction-ratio', help='Ratio (between 0 and 1) of background motion needed to accept object',
                        default=0.25, metavar='RATIO', type=float)
    parser.add_argument('--enable-background-masking', help='Enable masking of camera view with background subtraction',
                        default=False, action='store_true')
    parser.add_argument('--interframe-delay', help='Milliseconds of delay to allow between each video frame; drop frames not processed in time',
                        default=None, metavar='MSECS', type=int)
    parser.add_argument('--object-detector-skip-frames', help='Static number of frames to skip in between invocations of object detector',
                        default=None, metavar='N', type=int)
    parser.add_argument('--max-queue-size', help='Maximum size of the queues that communicate frames between co-routines internally.',
                        default=5, metavar='N', type=int)
    parser.add_argument('--log', help='Log state of parameters in given file as JSON',
                        default=None, metavar='FILE')
    parser.add_argument('--restore-from-log', help='Restore parameters from last line of log file, if present',
                        default=False, action='store_true')
    parser.add_argument('--object-annotation', help='The category of information to show with each detected object (options: ID, LABEL, NONE).',
                        default='LABEL', metavar='CATEGORY', choices=['ID','id','LABEL','label','NONE','none'])
    parser.add_argument('--cpu-temp-file', help='Specify system file to read CPU temperature from',
                        default=None, metavar='FILE')
    parser.add_argument('--cpu-freq-file', help='Specify system file to read CPU frequency from',
                        default=None, metavar='FILE')
    parser.add_argument('--disable-powersaving', help='Disable the insertion of powersaving delay into the pipeline.',
                        default=False, action='store_true')
    parser.add_argument('--powersave-delay-increment', help='How much delay in milliseconds to add for every frame observed with zero objects in it.',
                        default=10, metavar='MSEC',type=int)
    parser.add_argument('--powersave-delay-maximum', help='Maximum amount of pipeline delay in milliseconds to wait when observing a scene with no objects in it.',
                        default=500, metavar='MSEC',type=int)
    parser.add_argument('--focallength-mm', help='Focal length in mm', default=None, metavar='MM',type=float)
    parser.add_argument('--sensor-width-mm', help='Sensor width in mm', default=None, metavar='MM',type=float)
    parser.add_argument('--sensor-height-mm', help='Sensor height in mm', default=None, metavar='MM',type=float)
    parser.add_argument('--elevation-m', help='Elevation of camera in m', default=None, metavar='M',type=float)
    parser.add_argument('--tilt-deg', help='Camera tilt (straight down is 0 degrees)', default=None, metavar='DEG',type=float)
    parser.add_argument('--roll-deg', help='Camera roll (horizontal is 0 degrees)', default=0.0, metavar='DEG',type=float)
    parser.add_argument('--topdownview-size-m', help='X,Y in metres describing top-down view of area covered by camera.', default=None, metavar='X,Y')
    parser.add_argument('--3d', help='Toggle 3-D perspective unprojection transformation', default=False, action='store_true',dest='three_d')
    parser.add_argument('--raw-output', help='Only output raw video frames without any further drawing on them.', default=False, action='store_true')
    parser.add_argument('--centernet-mobilenet', help='special mode for centernet-mobilenet.', default=False, action='store_true')
    args = parser.parse_args(args=argv)

    if args.deepsorthome is None:
        args.deepsorthome = basedir
    args.basedir = basedir

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

# signal handlers
async def cancel_and_shutdown(loop, pipeline, sig=None):
    if sig is not None: print('Signal received: {}'.format(sig.name))
    pipeline.shutdown()

def make_exception_handler(pipeline):
    def handler(loop, context):
        msg = context.get("message", "no message")
        exc = context.get("exception", None)
        print('handle_exception: {}'.format(msg))
        if exc is not None:
            for s in traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__):
                print(s)
        asyncio.ensure_future(cancel_and_shutdown(loop, pipeline))
    return handler

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

    signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
    for s in signals:
        loop.add_signal_handler(s, lambda s=s: asyncio.create_task(cancel_and_shutdown(loop, pipeline, s)))
    loop.set_exception_handler(make_exception_handler(pipeline))

    # Kickstart the main pipeline
    asyncio.ensure_future(pipeline.start())
    asyncio.ensure_future(pipeline.periodic_heartbeat())

@webapp.after_serving
async def shutdown():
    global cmdserver
    if cmdserver:
        cmdserver.close()
    cmdserver = None


async def start_webapp():
    global shutdown_event
    shutdown_event = asyncio.Event()
    await serve(webapp, Config(), shutdown_trigger=shutdown_event.wait)

if __name__ == '__main__':
    uvloop.install()
    try:
        asyncio.run(start_webapp())
    except concurrent.futures.CancelledError:
        pass

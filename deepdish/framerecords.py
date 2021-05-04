from typing import Dict
import sys
import os
import numpy as np
from copy import copy
import deep_sort.track
import xml.etree.ElementTree as ET

def debug(*s,**kw):
    pass # print(*s,**kw)

class FrameRecord:
    def __init__(self, tlbr, label_id, order=None):
        self.tlbr = tlbr
        self.label_id = label_id
        self.order = order

class AnnotationRecord(FrameRecord):
    def __init__(self, annot_track_id: int, lbl, det_lbl_id: int, tlbr, outside: bool, occluded: bool, keyframe: bool, z_order: int, order=None):
        self.annotation_track_id = annot_track_id
        self.annotation_label = lbl
        self.is_outside = outside
        self.is_occluded = occluded
        self.is_keyframe = keyframe
        self.z_order = z_order
        self.tentative_matches: Dict[int, TentativeRecord] = {}
        self.score = 1.0
        super().__init__(tlbr, det_lbl_id, order=order)

class TentativeRecord(FrameRecord):
    def __init__(self, tlbr, label_id, score, order=None):
        self.score = score
        super().__init__(tlbr, label_id, order=order)

# compute overlapping percentage of the two framerecords based on their coordinates
def overlap(a: FrameRecord, b: FrameRecord):
    ax1, ay1, ax2, ay2 = list(a.tlbr)
    bx1, by1, bx2, by2 = list(b.tlbr)
    actual = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    most_possible = min(abs(ax2 - ax1) * abs(ay2 - ay1), abs(bx2 - bx1) * abs(by2 - by1))
    return actual / most_possible

class FrameRecords:
    def __init__(self, detector_id_to_labelname, overlap_threshold=0.9, override_tentative_detections=True, minimum_track_frames=3):
        self.frames = {}
        self.labels = {}
        self.detector_id_to_labelname = detector_id_to_labelname
        self.detector_labelname_to_id = {v: k for k, v in detector_id_to_labelname.items()}
        self.overlap_threshold = overlap_threshold
        self.override_tentative_detections = override_tentative_detections
        self.minimum_track_frames = minimum_track_frames

    def add_annotation_label_info(self, annotlabelname, detectorlabelid, annotlabelcolor):
        self.labels[annotlabelname] = {'detector_id': detectorlabelid, 'color': annotlabelcolor}

    def add_annotated_track(self, frame, annot_track_id, lbl, pts, outside, occluded, keyframe, z_order):
        det_lbl_id = self.labels[lbl]['detector_id']
        rec = AnnotationRecord(annot_track_id, lbl, det_lbl_id, pts, outside, occluded, keyframe, z_order)
        if frame not in self.frames:
            self.frames[frame] = []
        self.frames[frame].append(rec)

    def process_boxes(self, frame, boxes_in, labelnames_in, scores_in):
        tentatives = []
        for i, (tlwh, lblname, score) in enumerate(zip(boxes_in, labelnames_in, scores_in)):
            # boxes come from object detector in 'top,left,width,height' format
            tlbr = copy(tlwh)
            # tlbr in 'top,left,bottom,right' format
            tlbr[2:] = tlbr[:2] + tlbr[2:]
            # labels come in as strings
            lbl = self.detector_labelname_to_id[lblname]
            #debug('tentative {} {} {}'.format(tlbr,lbl,score))
            tentatives.append(TentativeRecord(tlbr, lbl, score, order=i))
        i = len(tentatives)
        if frame not in self.frames: self.frames[frame] = []

        # divide results into pools
        annotation_overlaps_tentative = []
        annotation_without_tentative = []
        tentative_without_annotation = copy(tentatives)
        unhandled_annotation = []

        for rec in self.frames[frame]:
            if isinstance(rec, AnnotationRecord):
                # Compare annotation record against tentatives
                overlap_found = False
                for ti, tentative in enumerate(tentative_without_annotation):
                    if overlap(rec, tentative) >= self.overlap_threshold:
                        if rec.label_id == tentative.label_id or rec.label_id is None:
                            # It is likely that the tentative detection is the same as the recorded annotation
                            # If rec.label_id is None then the annotation label may be different than the detector label
                            rec.tentative_matches[frame] = copy(tentative)
                            annotation_overlaps_tentative.append(rec)
                            debug('annotation_overlaps_tentative ti={} tlbr={} lbl={} overlap={}'.format(ti, rec.tlbr, rec.label_id, overlap(rec, tentative)))
                            overlap_found = True
                            del tentative_without_annotation[ti]
                            break

                if not overlap_found and rec.label_id is not None:
                    # No overlapping tentative found, so add the annotation info to tentatives
                    annotation_without_tentative.append(rec)
                    debug('annotation_without_tentative tlbr={} lbl={} ({}) score=1.0'.format(rec.tlbr, rec.label_id, rec.annotation_label))
                elif rec.label_id is None:
                    debug("unhandled_annotation with given label='{}' tlbr={}".format(rec.annotation_label, rec.tlbr))
                    unhandled_annotation.append(rec)

        # the following should be processed through the encoder and tracker:
        resultrecs = [*annotation_overlaps_tentative, *tentative_without_annotation, *annotation_without_tentative]

        boxes_out, labels_out, scores_out = [], [], []
        for i, rec in enumerate(resultrecs):
            rec.order = i
            tlwh = copy(rec.tlbr)
            tlwh[2:] = tlwh[2:] - tlwh[:2]
            boxes_out.append(tlwh)
            labels_out.append(self.detector_id_to_labelname[rec.label_id])
            scores_out.append(rec.score)

        resultrecs.extend(unhandled_annotation) # keep track of these for the future
        self.frames[frame] = resultrecs

        return boxes_out, labels_out, scores_out

    def process_detections(self, frame, detections):
        for det, rec in zip(detections, self.frames[frame]):
            rec.detection = det
            det.record = rec
        return detections

    def process_tracking(self, frame, tracker, tracks=None):
        if tracks is None: tracks = tracker.tracks
        annotation_track_db = {} # indexed by annotation_track_id
        for t in tracks:
            ndets = len(t.detections)
            # number of detections 'with record' entry
            ndetswithrec = len([d for d in t.detections if hasattr(d, 'record')])
            debug('Track id={} label={} ndets={} withrec={} time_since_update={}'.format(t.track_id,t.get_label(),ndets,ndetswithrec,t.time_since_update), end='\n  ')
            track_ids = set()
            for d in t.detections:
                if hasattr(d, 'record') and isinstance(d.record, AnnotationRecord):
                    track_ids.add(d.record.annotation_track_id)
            if len(track_ids) > 1:
                debug('Unexpected multiple track_ids: {}'.format(track_ids), end='\n  ')
            elif len(track_ids) == 1:
                # Retrieve update from frame records
                i = track_ids.pop()
                r = ([r for r in self.frames[frame] if isinstance(r, AnnotationRecord) and r.annotation_track_id == i] or (None,))[0]
                # r can be used to extend the track
                if r is not None:
                    db_entry = {'tracker_id': t.track_id, 'ndets_with_rec': ndetswithrec}
                    debug('found record annotation_track_id={} db_entry={}'.format(i,db_entry), end='\n  ')
                    # note down all tracks that correspond to
                    # particular annotation_track_ids because there
                    # can be duplicates
                    if i not in annotation_track_db: annotation_track_db[i] = []
                    annotation_track_db[i].append(db_entry)
                    if t.time_since_update > 0:
                        t.update(tracker.kf, r.detection)
                        t.state = deep_sort.track.TrackState.Confirmed
                        t.time_since_update = 0
            for d in t.detections:
                if hasattr(d, 'record') and isinstance(d.record, AnnotationRecord):
                    debug('{}'.format(d.record.annotation_track_id), end=',')
                if hasattr(d, 'record'):
                    d.record.track = t

            debug()

        tracks_to_delete = set()
        for i, e in annotation_track_db.items():
            max_entry = max(e, key=lambda x: x['ndets_with_rec'])
            debug('annotation_track_db[{}] = {} (max: {})'.format(i, e, max_entry))
            for x in e:
                # delete duplicate tracks that are tracing the same
                # AnnotationRecord (this can happen because detections
                # generated from AnnotationRecords get fed into the
                # tracker even if the tracker loses the original
                # track)
                if x['ndets_with_rec'] < max_entry['ndets_with_rec']:
                    tracks_to_delete.add(x['tracker_id'])

        debug('tracks_to_delete = {}'.format(tracks_to_delete))
        tracks_out = [t for t in tracks if t.track_id not in tracks_to_delete]
        return tracks_out

    def xml_output(self, meta=None):
        annotations = ET.Element('annotations')
        version = ET.SubElement(annotations, 'version')
        version.text = '1.1'
        if meta is not None:
            annotations.append(meta)
        annot_track_db = {}
        new_track_db = {}
        for frame, recs in self.frames.items():
            for rec in recs:
                if hasattr(rec, 'annotation_track_id'):
                    i = rec.annotation_track_id
                    if i not in annot_track_db: annot_track_db[i] = {}
                    annot_track_db[i][frame] = rec
                elif hasattr(rec, 'track'):
                    i = rec.track.track_id
                    if i not in new_track_db: new_track_db[i] = {}
                    new_track_db[i][frame] = rec

        # first create elements for known existing annotated tracks
        max_annot_track_id = 0
        for i, framedb in sorted(annot_track_db.items()):
            if i > max_annot_track_id: max_annot_track_id = i
            track = ET.SubElement(annotations, 'track', attrib={'id': str(i), 'source': 'manual'})
            for frame, rec in sorted(framedb.items()):
                #     <box frame="6" outside="0" occluded="1" keyframe="1" xtl="1724.28" ytl="996.09" xbr="1855.16" ybr="1080.00" z_order="0">
                ET.SubElement(track, 'box', attrib={'frame': str(frame),
                                                    'occluded': '1' if rec.is_occluded else '0',
                                                    'outside': '1' if rec.is_outside else '0',
                                                    'keyframe': '1' if rec.is_keyframe else '0',
                                                    'z_order': str(rec.z_order),
                                                    'xtl': str(rec.tlbr[0]),
                                                    'ytl': str(rec.tlbr[1]),
                                                    'xbr': str(rec.tlbr[2]),
                                                    'ybr': str(rec.tlbr[3])})
                label = self.detector_id_to_labelname[rec.label_id] # should be the same for every rec
            track.set('label', label)

        # append elements for new tracks
        max_annot_track_id += 1
        for _, framedb in sorted(new_track_db.items()):
            if len(framedb) >= self.minimum_track_frames:
                # use new track id labels greater than existing track id labels
                i = max_annot_track_id
                max_annot_track_id += 1
                track = ET.SubElement(annotations, 'track', attrib={'id': str(i), 'source': 'automatic'})
                # figure out the label of this track by max occurrence
                label_ids = {}
                for frame, rec in sorted(framedb.items()):
                    #     <box frame="6" outside="0" occluded="1" keyframe="1" xtl="1724.28" ytl="996.09" xbr="1855.16" ybr="1080.00" z_order="0">
                    if rec.label_id not in label_ids: label_ids[rec.label_id] = 0
                    label_ids[rec.label_id] += 1
                    box = ET.SubElement(track, 'box', attrib={'frame': str(frame),
                                                              'occluded': '0',
                                                              'outside': '0',
                                                              'keyframe': '1',
                                                              'z_order': '0',
                                                              'xtl': str(rec.tlbr[0]),
                                                              'ytl': str(rec.tlbr[1]),
                                                              'xbr': str(rec.tlbr[2]),
                                                              'ybr': str(rec.tlbr[3])})
                box.set('outside', '1') # set last entry to 'outside=1'
                # label with most occurrences is 'the label'
                label = self.detector_id_to_labelname[max(label_ids, key=label_ids.get)]
                track.set('label', label)
        tree = ET.ElementTree(annotations)
        self.indent(tree)
        # with os.fdopen(sys.stdout.fileno(), "wb", closefd=False) as stdout:
        #     # Output tree to stdout
        #     tree.write(stdout, xml_declaration=True, encoding='utf-8', short_empty_elements=False)
        #     stdout.flush()
        return tree



    # from python 3.9
    def indent(self, tree, space="  ", level=0):
        """Indent an XML document by inserting newlines and indentation space
        after elements.
        *tree* is the ElementTree or Element to modify.  The (root) element
        itself will not be changed, but the tail text of all elements in its
        subtree will be adapted.
        *space* is the whitespace to insert for each indentation level, two
        space characters by default.
        *level* is the initial indentation level. Setting this to a higher
        value than 0 can be used for indenting subtrees that are more deeply
        nested inside of a document.
        """
        if isinstance(tree, ET.ElementTree):
            tree = tree.getroot()
        if level < 0:
            raise ValueError(f"Initial indentation level must be >= 0, got {level}")
        if not len(tree):
            return

        # Reduce the memory consumption by reusing indentation strings.
        indentations = ["\n" + level * space]

        def _indent_children(elem, level):
            # Start a new indentation level for the first child.
            child_level = level + 1
            try:
                child_indentation = indentations[child_level]
            except IndexError:
                child_indentation = indentations[level] + space
                indentations.append(child_indentation)

            if not elem.text or not elem.text.strip():
                elem.text = child_indentation

            for child in elem:
                if len(child):
                    _indent_children(child, child_level)
                if not child.tail or not child.tail.strip():
                    child.tail = child_indentation

            # Dedent after the last child by overwriting the previous indentation.
            if not child.tail.strip():
                child.tail = indentations[level]

        _indent_children(tree, 0)

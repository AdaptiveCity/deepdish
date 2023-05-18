import glob
import os
import cv2
import time
import face_detection
import csv
import pandas as pd
from motrackers import SORT
import numpy as np

def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)

def reformat_faces(bboxes):
	res = []
	for bbox in bboxes:
		x0, y0, x1, y1 = bbox
		res.append((x0, y0, (x1-x0), (y1-y0)))
	return res

def main():
    directory = os.path.dirname(__file__)
    capture = cv2.VideoCapture(os.path.join(directory, "MOT17-11.mp4"))
    
    face_detector = face_detection.build_detector(
        "DSFDDetector",
        max_resolution=1080
    )
    
    facemulTracker = SORT()
    
    dfdata = []
    
    header = ['framenum', 'x', 'y', 'w', 'h', 'id']

    framenum = 0
    
    num = 0
    outwriter = cv2.VideoWriter('MOT17-11_output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (100,100))

    while True:

        result, image = capture.read()


        channels = 1 if len(image.shape) == 2 else image.shape[2]
        if channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        height, width, _ = image.shape
        
        if(num == 0):
        	outwriter = cv2.VideoWriter('MOT17-11_output.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, (width, height))
        	num = num + 1


        print("Processing:", framenum)
        t = time.time()
        dets = face_detector.detect(
            image[:, :, ::-1]
        )[:, :4]
        print(f"Detection time: {time.time()- t:.3f}")
        faces = reformat_faces(dets)
        faces = faces if faces is not None else []
        
        ofaces =[]
		
        for face in faces:
            box = list(map(int, face[:4]))
            ofaces.append(box)
        
        num_faces = len(ofaces)
        olist = np.array(ofaces)
        conf = np.ones(num_faces)
        cla = np.zeros(num_faces)
        
        output_tracks = facemulTracker.update(olist, conf, cla)
        print(output_tracks)
        
        for track in output_tracks:
        	frame, idnum, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
        	
        	box = (bb_left, bb_top, bb_width, bb_height)
        	color = (0, 0, 255)
        	thickness = 2
        	cv2.rectangle(image, box, color, thickness, cv2.LINE_AA)
        	
        	centroid = (int((bb_left)), int((bb_top + bb_height)))
        	
        	text = str(idnum)
        	
        	#cv2.putText(image, text, centroid, text, align ="right", fill = (0,0,0))
        	
        	font = cv2.FONT_HERSHEY_SIMPLEX
        	fontScale = 0.5
        	
        	print(box)
        	print(centroid)
        	
        	cv2.putText(image, text, centroid, font, fontScale, (0, 255, 0), thickness, cv2.LINE_AA)
        	
        	dfdata.append([framenum, box[0], box[1], box[2], box[3], idnum])
        	
        outwriter.write(image)
        framenum = framenum + 1

        cv2.imshow("face detection", image)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break

    df = pd.DataFrame(dfdata, columns=header)
    df.to_csv('MOT17-11_ground_truth.csv', index=False)
    outwriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
import cv2
import pandas as pd
import os

import motmetrics as mm
import numpy as np
		
def main():
	directory = os.path.dirname(__file__)
	gtpath = os.path.join(directory, "MOT17-05_ground_truth.csv")
	predpath = os.path.join(directory, "test2.csv")
	
	gtdf = pd.read_csv(gtpath)
	prdf = pd.read_csv(predpath)
	
	totalg = len(gtdf)
	totalp = len(prdf)
	
	acc = mm.MOTAccumulator(auto_id=True)
	
	dfdata = []
	
	header = ['precision', 'recall']
	TP = 0
	FP = 0
	
	print(prdf['framenum'].max())
	
	num = min(prdf['framenum'].max(), gtdf['framenum'].max())
	
	fdata = []
	
	header = ['precision', 'recall']
	start = True
	det = 0
	
	for i in range(gtdf['framenum'].max()):
		igt = gtdf.loc[gtdf['framenum'] == (i-1)]
		ipr = prdf.loc[prdf['framenum'] == (i)]
		
		prlist = []
		gtlist = []
		
		prid = []
		gtid = []
		
		for prindex, prrow in ipr.iterrows():
			prbox = [prrow['x'], prrow['y'], prrow['x'] + prrow['w'], prrow['y'] +prrow['h']]
			
			prlist.append(prbox)
			prid.append(prrow['id'])
			
		for gtindex, gtrow in igt.iterrows():
			gtbox = [gtrow['x'], gtrow['y'], gtrow['x'] + gtrow['w'], gtrow['y'] +gtrow['h']]
			
			gtlist.append(gtbox)
			gtid.append(gtrow['id'])
			
		a = np.array(prlist)
		b = np.array(gtlist)
		
		distance_matrix = mm.distances.iou_matrix(b, a, max_iou=0.5)
		print(distance_matrix)
		
		acc.update(gtid, prid, [distance_matrix])
		
		print("BOXES")
		print(b)
		print(a)
		
		print("IDS")
		print(gtid)
		print(prid)
		
		mh = mm.metrics.create()
		
		report = mh.compute(acc, metrics=['num_frames', 'mota','motp', 'num_detections', 'num_misses','num_false_positives','num_switches', 'precision', 'recall'], name='acc')
		print(report)
		if(start):
			start = False
		else:
			det = report['num_detections']['acc']
			
			fp = report['num_false_positives']['acc']
			
			recall = det/totalg
			precision = det/(det+fp)
		
			dfdata.append([precision, recall])
	
	print("FINAL VALUES")
	print(report)
	print(totalg)
	print(precision)
	print(recall)
	dfdata.append([0,1.0])
	df = pd.DataFrame(dfdata, columns=header)
	df.to_csv('Testpr.csv', index=False)
	
if __name__ == '__main__':
    main()
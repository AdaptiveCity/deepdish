import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

def getMaxPrecision(startingval, rows):
	endingval = startingval+0.1
	dr = rows.loc[rows['recall'] <= endingval].loc[rows['recall'] > startingval]
	if(len(dr) == 0):
		return 0
	return dr['precision'].max()

def graphPR(input_file, graph_title, output_file):
	directory = os.path.dirname(__file__)
	path = os.path.join(directory, input_file)
	
	df = pd.read_csv(path)
	
	epoints = []
	flag = True
	
	total = 0
	saved = 0
	for i in range(10):
		j = (9-i) * 0.1
		maxprecision = max(getMaxPrecision(j, df), saved)
		if(flag and maxprecision >= 0):
			epoints.append([j+0.2, 0])
			epoints.append([j+0.1, 0])
			if(maxprecision > 0):
				flag = False
		if(maxprecision >= 0):
			epoints.append([j+0.1, maxprecision])
			epoints.append([j, maxprecision])
			saved = maxprecision
		total = total + (maxprecision * 0.1)
	
	points = []
	for index, row in df.iterrows():
		points.append([row['recall'], row['precision']])
	
	x,y = np.array(points).T
	font = {'family' : 'Times', 'weight' : 'bold', 'size'   : 16}
	plt.rc('font', **font)
	plt.scatter(x,y, s = 10, label = 'original')
	ex,ey = np.array(epoints).T
	plt.plot(ex,ey, color = 'k', label = 'interpolated')
	plt.title(graph_title)
	plt.xlabel('recall')
	plt.ylabel('precision')
	plt.legend(loc="lower left")
	plt.savefig(output_file)
	plt.show()
	
	print(total)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--Input", help = "Input File Name")
	parser.add_argument("-t", "--Title", help = "Graph Title")
	parser.add_argument("-o", "--Output", help = "Output File Name")
	
	args = parser.parse_args()
	graphPR(args.Input, args.Title, args.Output)
import csv
import numpy as np
import argparse
import sys
import os
import cv2

def error(gt, result):
	# accard similarity: which is the ratio between the intersection 
	# and union of the segmented results and the ground truth
	# J= (A intersect B) / (A union B) 
	gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
	_, gt = cv2.threshold(gt, 128, 255,cv2.THRESH_BINARY)
	
	intersect = cv2.bitwise_and(gt, result)
	union = cv2.bitwise_or(gt, result)
	whitesI = np.sum(intersect == 255)
	whitesU = np.sum(union == 255)
	e = whitesI/whitesU
	return e

def main(args=None):
	files = []
	labels = []
	# r=root, d=directories, f = files
	for r, _, f in os.walk(os.path.join("dataset", "images")):
		for file in f:
			files.append(os.path.join(r, file))

	for r, _, f in os.walk(os.path.join("dataset", "labels")):
		for label in f:
			labels.append(os.path.join(r, label))			

	te = me = ge = tre = 0
	for idx, f in enumerate(files):
		#print(f)
		image = cv2.imread(f)
		#print(labels[idx])
		#print(labels)
		label = cv2.imread(labels[idx])
		grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		_, threshold = cv2.threshold(grayscale, 128,255,cv2.THRESH_BINARY_INV)
		mean = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,15,2)
		gaussian = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,2)
		 
		while(1):
			cv2.imshow('Original', cv2.resize(image, (250, 250)))
			cv2.imshow('Grayscale', cv2.resize(grayscale, (250, 250)))
			cv2.imshow('Threshold (Binary)', cv2.resize(threshold, (250, 250)))
			cv2.imshow('Mean Adaptive Threshold', cv2.resize(mean, (250, 250)))
			cv2.imshow('Gaussian Adaptive Threshold', cv2.resize(gaussian, (250, 250)))
			cv2.imshow("GT", cv2.resize(label, (250, 250)))

			k = cv2.waitKey(33)
			if k==27:    # Esc key to stop
				exit(0)
			elif k==-1:  # normally -1 returned,so don't print it
				continue
			elif k==32:
				te += error(label, threshold)
				me += error(label, mean)
				ge += error(label, gaussian)

				label2 = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
				_, label2 = cv2.threshold(label2, 128, 255,cv2.THRESH_BINARY)
				tre += error(label, label2)
				print("TH:", error(label, threshold), "  \tME:", error(label, mean), "  \tGA:", error(label, gaussian), "  \tGT:", error(label, label2))
				break
			else:
				print(k)
				
			cv2.destroyAllWindows()

	print("Averaged:")
	print("TH:", te/68, "  \tME:", me/68, "  \tGA", ge/68, "\t\tGT", tre/68)	

if __name__== "__main__":
	main()
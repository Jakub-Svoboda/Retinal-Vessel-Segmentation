import csv
import numpy as np
import argparse
import sys
import os
import cv2

def threshold(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, image = cv2.threshold(image,50,255,cv2.THRESH_BINARY)
	cv2.imshow("image",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def main(args=None):
	path = os.path.join("dataset", "images")
	files = []
	# r=root, d=directories, f = files
	for r, d, f in os.walk(path):
		for file in f:
				files.append(os.path.join(r, file))

	for f in files:
		print(f)
		img = cv2.imread(f)
		threshold(img)
		#break #!!!!!!!!!!!!!!!!!!!!!!!!

if __name__== "__main__":
	main()
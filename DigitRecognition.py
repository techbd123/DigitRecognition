from __future__ import print_function
import sys
import numpy as numpy
import cv2

def ProcessPixel(image):
	h,w,c=image.shape
	imageBitMap=""
	for x in range(0,h):
		for y in range(0,w):
			val=int(image[x,y,2])
			val+=image[x,y,1]
			val+=image[x,y,0]
			if val*2<765:
				imageBitMap+="1"
			else:
				imageBitMap+="0"
		imageBitMap+="\n"
	return imageBitMap

def main():
	image=cv2.imread("E1_105.jpg")
	imageBitMap=ProcessPixel(image)
	print(imageBitMap,end="")
	return 0

main()
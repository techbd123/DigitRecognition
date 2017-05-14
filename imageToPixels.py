from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import numpy as np
import scipy as sp
from scipy import misc
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def ProcessPixel(image):
	h,w,c=image.shape
	imageBitMap=""
	pixel=[]
	for x in range(0,h):
		for y in range(0,w):
			val=int(image[x,y,2])
			val+=image[x,y,1]
			val+=image[x,y,0]
			if val*2<765:
				imageBitMap+="1"
				pixel.append(1)
			else:
				imageBitMap+="0"
				pixel.append(0)
		imageBitMap+="\n"
	return pixel,imageBitMap

def main():
	image=misc.imread(sys.argv[1])
	pixels,imageBitMap=ProcessPixel(image)
	print(imageBitMap,end="")
	return 0

main()
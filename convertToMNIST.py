from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import argparse
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import scipy
from array import *
from random import shuffle


def LabelToByte(label):
	if label[0]=='E' :
		return int(label[1])-int('0')
	return int(label[1])-int('0')+10


# Load from and save to
Names = [['./dataset/trainset','./dataset/train'], ['./dataset/testset','./dataset/test']]

for name in Names:
	
	data_image = array('B')
	data_label = array('B')
	FileList = []
	for filename in os.listdir(name[0]):
		if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
			filename=name[0]+'/'+filename;
			FileList.append(filename)

	#shuffle(FileList) # Usefull for further segmenting the validation set

	for filename in FileList:
		print(filename)
		label=filename.split('/')[3]
		label=label[0]+label[1]
		print(label)

		image=misc.imread(filename)
		h,w,c=image.shape
		for x in range(h):
			for y in range(w):
				val=float(image[x,y,2])
				val+=image[x,y,1]
				val+=image[x,y,0]
				val/=3
				data_image.append(int(val))

		data_label.append(LabelToByte(label)) # labels start (one unsigned byte each)
		print(LabelToByte(label))
	
	hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
	print(hexval)

	# header for label array

	header = array('B')
	header.extend([0,0,8,1,0,0])
	print(header)
	header.append(int('0x'+hexval[2:][:2],16))
	print(header)
	header.append(int('0x'+hexval[2:][2:],16))
	print(header)
	
	data_label = header + data_label

	# additional header for images array
	
	if max([h,w]) <= 100:
		header.extend([0,0,0,h,0,0,0,w])
	else:
		raise ValueError('Image exceeds maximum size: 100x100 pixels');

	header[3] = 3 # Changing MSB for image data (0x00000803)
	print(header)
	
	data_image = header + data_image

	output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
	data_image.tofile(output_file)
	output_file.close()

	output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
	data_label.tofile(output_file)
	output_file.close()

# gzip resulting files

for name in Names:
	os.system('gzip '+name[1]+'-images-idx3-ubyte')
	os.system('gzip '+name[1]+'-labels-idx1-ubyte')
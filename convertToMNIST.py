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
import functions as fun
from array import *
from random import shuffle

Names = [['./dataset/trainset','./dataset/train'], ['./dataset/testset','./dataset/test']]

for name in Names:
	
	data_image = array('B')
	data_label = array('B')
	FileList = []
	for filename in os.listdir(name[0]):
		if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
			filepath=name[0]+'/'+filename;
			FileList.append([filepath,filename])

	# Shuffling is done so that the exact dataset is NOT produced
	shuffle(FileList)

	height,width,fnum=0,0,0
	for file in FileList:
		filepath=file[0]
		filename=file[1]
		label=filename[0]+filename[1]

		image=misc.imread(filepath)
		h,w,c=image.shape
		height=max([height,h])
		width=max([width,w])
		if h!=100 or w!=100:
			print('The size of image '+filepath+' is not of 100x100 pixels. So discard it!')
			continue
		print(filepath)
		print(label)
		for x in range(h):
			for y in range(w):
				val=float(image[x,y,2])
				val+=image[x,y,1]
				val+=image[x,y,0]
				val/=3
				data_image.append(int(val))

		data_label.append(fun.LabelToValue(label))
		fnum+=1
	#	print(LabelToByte(label))
	
	hexval = "{0:#0{1}x}".format(fnum,6)
#	print(hexval)

	# header for label array

	header = array('B')
	header.extend([0,0,8,1,0,0])
#	print(header)
	header.append(int('0x'+hexval[2:][:2],16))
#	print(header)
	header.append(int('0x'+hexval[2:][2:],16))
#	print(header)
	
	data_label = header + data_label

	# header for images array
	
	if max([height,width]) <= 100:
		header.extend([0,0,0,h,0,0,0,w])
	else:
		raise ValueError('Image exceeds maximum size: 100x100 pixels');

	header[3] = 3 # Changing MSB for image data (0x00000803)
#	print(header)
	
	data_image = header + data_image

	output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
	data_image.tofile(output_file)
	output_file.close()

	output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
	data_label.tofile(output_file)
	output_file.close()

# gzip

for name in Names:
	os.system('gzip -f '+name[1]+'-images-idx3-ubyte')
	os.system('gzip -f '+name[1]+'-labels-idx1-ubyte')

print("Conversion to MNIST format is successful!")
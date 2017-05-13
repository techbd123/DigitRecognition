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

def LabelToValue(label):
	'''
	E0 = 0
	E1 = 1
	E2 = 2
	E3 = 3
	E4 = 4
	E5 = 5
	E6 = 6
	E7 = 7
	E8 = 8
	E9 = 9
	B0 = 10
	B1 = 11
	B2 = 12
	B3 = 13
	B4 = 14
	B5 = 15
	B6 = 16
	B7 = 17
	B8 = 18
	B9 = 19
	'''
	if label[0]=='E' :
		return int(label[1])-int('0')
	return int(label[1])-int('0')+10


def ValueToLabel(value):
	if value<10:
		return 'E'+str(value)
	return 'B'+str(value-10)

def DisplayDigit(images,height,width,labels,index):
    label=labels[index].argmax(axis=0)
    image=images[index].reshape([height,width])
    plt.title('Sample #%d  Label = %s' % (index,ValueToLabel(label)))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
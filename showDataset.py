from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mnist
import functions as fun

height=100
width=100
num_input_pixels=height*width
num_classes=20

data=mnist.read_data_sets("dataset",one_hot=True,num_classes=num_classes)

for i in range(data.train.images.shape[0]):
	fun.DisplayDigit(data.test.images,height,width,data.test.labels,i)
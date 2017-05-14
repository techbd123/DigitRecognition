from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import numpy as np
import tensorflow as tf
import mnist
import matplotlib.pyplot as plt
import scipy
import random as ran
import functions as fun

height=100
width=100
num_input_pixels=height*width
num_classes=20
num_batches=100
num_iterations=10000

data = mnist.read_data_sets("dataset/",one_hot=True,num_classes=num_classes)

# Create the model
x = tf.placeholder(tf.float32, [None, num_input_pixels])
W = tf.Variable(tf.zeros([num_input_pixels,num_classes]))
b = tf.Variable(tf.zeros([num_classes]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, num_classes])

# The raw formulation of cross-entropy,
tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),reduction_indices=[1]))
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Train
print('Training started!')

for _ in range(num_iterations):
  batch_x, batch_y = data.train.next_batch(num_batches)
  sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})

print('Training finished!')


# Test trained model
print('Testing started!')

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy = '+str(sess.run(accuracy, feed_dict={x: data.test.images,y_: data.test.labels})*100))

print('Testing finished!')

fun.DisplayDigit(data.test.images,height,width,data.test.labels,ran.randint(0,data.test.labels.shape[0]))
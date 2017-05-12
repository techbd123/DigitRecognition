from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import scipy
import random as ran

#digit = scipy.ndimage.imread("dataset/allDigitImages/E9_31.jpg")

#scipy.misc.imshow(digit)

mnist = input_data.read_data_sets("dataset/",one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 10000])
W = tf.Variable(tf.zeros([10000, 20]))
b = tf.Variable(tf.zeros([20]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 20])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(1)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels})*100)

def display_digit(num):
    print(batch_ys[num])
    label = batch_ys[num].argmax(axis=0)
    image = batch_xs[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

display_digit(ran.randint(0,batch_xs.shape[0]))
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
# Initial input

height=100
width=100
num_input_pixels=height*width
num_classes=20
num_batchSize=40

# Extract data
data = mnist.read_data_sets("dataset",one_hot=True,num_classes=num_classes)

learning_rate=0.001
training_epochs=40
display_step=1

# Neural Network parameters
num_hidden_1 = 4096 # 1st layer number of features (units)
num_hidden_2 = 4096 # 2nd layer number of features (units)

# tf Graph input
x = tf.placeholder(tf.float32, [None,num_input_pixels])

# tf Graph output
y = tf.placeholder(tf.float32, [None,num_classes])


# Showing Data
print('\nnum_input_pixels = '+str(num_input_pixels))
print('num_classes = '+str(num_classes))
print('num_batchSize = '+str(num_batchSize))
print('num_hidden_1 = '+str(num_hidden_1))
print('num_hidden_2 = '+str(num_hidden_2))
print('training_epochs = '+str(training_epochs))
print('learning_rate = '+str(learning_rate))
print('')

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer 1 with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer 2 with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights ={
    'h1': tf.Variable(tf.random_normal([num_input_pixels, num_hidden_1])),
    'h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    print('Training Started!')
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(data.train.num_examples/num_batchSize)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = data.train.next_batch(num_batchSize)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost],feed_dict={x: batch_x,y: batch_y})
            # Compute average loss
            avg_cost+= c/total_batch
        # Display cost per epoch step
        if epoch%display_step==0:
            print('Epoch:', '%04d' % (epoch+1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Training Finished!")

    # Test trained model
    print('Testing started!')
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	
	# Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: data.test.images, y: data.test.labels})*100)
    print('Testing finished!')

fun.DisplayDigit(data.test.images,height,width,data.test.labels,ran.randint(0,data.test.labels.shape[0]))
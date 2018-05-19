""" Neural Network.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function
import os
import time

# User your username on DAVINCI to create a directory for storing the MNIST
# dataset
user = os.environ['USER']

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/scratch/" + user + "/mnist/", one_hot=False)

import tensorflow as tf

# Parameters
learning_rate = 0.1
num_epochs = 20
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


# Define the neural network to take an input of 'batch_size' images, each with
# 'num_input' pixels with 2 hidden layers, each with 256 neurons.
def neural_net():
    x = tf.placeholder('float', shape = (batch_size, num_input), name = 'images')
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.layers.dense(x, n_hidden_1)
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.layers.dense(layer_2, num_classes)
    return x, out_layer


# Compute the error between the expected labels and the computed labels across
# the whole training dataset. Error is defined by the loss_op operation.
def compute_err(loss_op, image_placeholder, label_placeholder, images, labels):
    err = 0.0
    n_batches = 0
    start_batch = 0
    while start_batch + batch_size < mnist.train.images.shape[0]:
        err += sess.run(loss_op,
                feed_dict = {image_placeholder: images[start_batch:start_batch + batch_size, :],
                             label_placeholder: labels[start_batch:start_batch + batch_size]})
        start_batch += batch_size
        n_batches += 1
    return err / float(n_batches)

# Construct the TF graph model, returning the input placeholder as 'images' and
# the output as 'logits'.
images, logits = neural_net()

# Create a placeholder to store the expected values for each batch, for error
# calculation by the optimizer.
labels = tf.placeholder('uint8', shape = (batch_size), name = 'labels')

# Measure the loss between the computed logits and the expected labels
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(labels, dtype=tf.int32)))

# Configure an optimizer with the given learning rate
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# Ask the optimizer to minimize the model loss defined above
train_op = optimizer.minimize(loss_op,
                              global_step=tf.train.get_global_step())

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # Initialize all variables
    sess.run(init)

    # Iterate for the specified number of epochs
    start_training_time = time.time()
    for epoch in range(num_epochs):

        # Batch the inputs into batches of length 'batch_size' and pass them
        # through the optimizer batch-by-batch.
        start_batch = 0
        while start_batch + batch_size < mnist.train.images.shape[0]:
            sess.run(train_op, feed_dict = {images: mnist.train.images[start_batch:start_batch + batch_size, :],
                                            labels: mnist.train.labels[start_batch:start_batch + batch_size]})
            start_batch += batch_size

        # Compute and print the error of the current model on the training data
        err = compute_err(loss_op, images, labels, mnist.train.images,
                mnist.train.labels)
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(err))
    elapsed_training_time = time.time() - start_training_time

    # Evaluate the trained model on MNIST testing data
    pred_classes = tf.argmax(logits, axis = 1)
    computed_labels = []
    start_batch = 0
    while start_batch + batch_size < mnist.test.images.shape[0]:
        predictions = sess.run(pred_classes,
                feed_dict = {images: mnist.test.images[start_batch:start_batch + batch_size, :]})
        for p in predictions:
            computed_labels.append(p)
        start_batch += batch_size

    # Count how many of the labels computed by our model match the expected
    # labels
    count_same = 0
    for i in range(len(computed_labels)):
        if computed_labels[i] == mnist.test.labels[i]:
            count_same += 1
    print(count_same, '/', len(computed_labels), '=',
            100.0 * float(count_same) / float(len(computed_labels)), '%', ',',
            elapsed_training_time, 's to train')

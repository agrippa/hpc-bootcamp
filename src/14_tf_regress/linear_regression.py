'''
A linear regression learning algorithm example using TensorFlow library.

This example constructs a simple linear model of Y = W * X + b, using a gradient
descent optimizer to minize model error.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters

# The learning rate controls how large a step the optimizer can take through the
# parameter space while exploring for a minimima.
learning_rate = 0.4

# How many times to pass the training data through the model while updating
# trainable variables. We perform many epochs to give the optimizer a chance to
# minimize the model error.
training_epochs = 40000

# How often to display a summary of our model's current accuracy during training
display_step = 100

# Load our data from a binary file on disk
input_X = numpy.fromfile('X.bin').reshape((-1, 3))
input_Y = numpy.fromfile('Y.bin')

print('Loaded ' + str(len(input_X)) + ' samples')

# Split our data into 80% training data, 20% testing data
train_ratio = 0.8
n_samples = int(train_ratio * len(input_X))

# Training Data
train_X = input_X[:n_samples, :]
train_Y = input_Y[:n_samples]

# Model parameters. These placeholders are fed into the model by our application.
X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
Y = tf.placeholder("float")

# Model weights. These weights are initialized to a random number, and then
# tuned by the optimizer to improve model accuracy.
W1 = tf.Variable(rng.randn(), name="weight1")
W2 = tf.Variable(rng.randn(), name="weight2")

# Construct a linear model that matches the structure of the original 1D
# iterative averaging example: X1*W1 + X2*W2
#
# Note that while the structure is the same, it is up to Tensorflow to learn the
# weights of the equation.
pred = tf.add(tf.multiply(X1, W1),
              tf.multiply(X2, W2))

# Mean squared error measures the difference between the expected values for
# each sample and the value computed by the model.
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# Gradient descent, our optimizer for this problem.
# Note, minimize() knows to modify W and b because Variable objects are
# trainable by default
#
# This code configures the optimizer with a learning rate (i.e. how large its
# updates to the model variables can be), and then points it to the value we
# would like it to minimize: the cost of the model.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# An operation that initializes our weights (i.e. assigns their default values)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Show what weights our model is initialized with
    print("Initialized: W=", sess.run(W1), sess.run(W2))

    # Execute several training epochs
    for epoch in range(training_epochs):
        # Pass the training data through the optimizer, allowing it to update
        # the variables in our model to reduce 'cost'
        sess.run(optimizer, feed_dict={X1: train_X[:, 0],
                                       X2: train_X[:, 2],
                                       Y: train_Y})

        # Display logs every 'display_step' steps, with information on our
        # current model weights and cost.
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X1: train_X[:, 0], X2: train_X[:, 2], Y: train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W1), sess.run(W2))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X1: train_X[:, 0], X2: train_X[:, 2], Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W1), sess.run(W2), '\n')

    # Testing data, to validate the accuracy of our model against unseen samples.
    test_X = input_X[n_samples:, :]
    test_Y = input_Y[n_samples:]

    # Compute our cost/error against the testing samples
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X1: test_X[:, 0], X2: test_X[:, 2], Y: test_Y})  # same function as cost above
    print("Testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

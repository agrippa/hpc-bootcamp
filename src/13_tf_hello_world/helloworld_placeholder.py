'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Simple hello world using TensorFlow

# Create a placeholder op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the value contained in the
# placeholder op.
message = tf.placeholder(dtype = tf.string, name = 'message')

# Start tf session
sess = tf.Session()

# Run the op
print(sess.run(message, feed_dict = {message: 'Hello, TensorFlow!'}))

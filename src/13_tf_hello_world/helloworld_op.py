'''
HelloWorld example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf


# Create a constant integer node in the TF graph.
c = tf.constant(3)

# Create a placeholder integer node, allowing us to feed a value at execution
# time.
v = tf.placeholder(dtype = tf.int32, name = 'v')

# Add the constant value 3 and the value stored in 'v' together.
sum = tf.add(c, v)

# Start tf session
sess = tf.Session()

# Run the graph, performing the operation 3 + v, and print the output.
#
# Below, we use feed_dict to set 'v' to 5.
print('')
print('Computed the value = '  + str(sess.run(sum, feed_dict = {v: 5})))

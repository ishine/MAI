import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

#1 2 2 3
input1=tf.constant([True, False, False, True])
input1=tf.reshape(input1, [1, 2, 1, 2])
input2=tf.constant([False, False, True, True, False, True, True, True])
input2=tf.reshape(input2, [2, 2, 2])
op1 = tf.logical_and(input1, input2, name='greater')
print(op1.eval())

#constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['where'])
#
#with tf.gfile.FastGFile("greater.pb", mode='wb') as f:
#    f.write(constant_graph.SerializeToString());


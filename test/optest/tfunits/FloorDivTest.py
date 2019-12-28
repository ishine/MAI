import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

#1 2 2 3
input1=tf.reshape(tf.constant([0, -1, 2, 100, 10, -100, 10, 17]), [2, 2, 2])
input2=tf.reshape(tf.constant([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2])
op1 = tf.floor_div(input1, input2, name='floor_div')
print(op1.eval())

input1=tf.reshape(tf.constant([1, 2, 3, 4]), [1, 2, 1, 2])
input2=tf.reshape(tf.constant([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2])
op1 = tf.floor_div(input1, input2, name='floor_div')
print(op1.eval())

input1=tf.reshape(tf.constant([1]), [1])
input2=tf.reshape(tf.constant([1, 2, 3, 4, 5, 6, 7, 8]), [2, 2, 2])
op1 = tf.floor_div(input1, input2, name='floor_div')
print(op1.eval())

input1=tf.reshape(tf.constant([1, 2, 3, 4]), [2, 1, 2])
input2=tf.reshape(tf.constant([2]), [1])
op1 = tf.floor_div(input1, input2, name='floor_div')
print(op1.eval())
#constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['where'])
#
#with tf.gfile.FastGFile("greater.pb", mode='wb') as f:
#    f.write(constant_graph.SerializeToString());


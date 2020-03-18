import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

#2 2 3
input=tf.constant([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]])
#o0 = tf.reduce_sum(input)
#target=o0.eval();
#print(target)

#o1 = tf.reduce_sum(input, [-1,-1])
#target=o1.eval();
#print(target)

o2 = tf.reduce_sum(input, [1,0], name="reduce_o2")
target=o2.eval();
print(target.shape)
print(target)

#constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['reduce_o2'])
#
#with tf.gfile.FastGFile("reduce.pb", mode='wb') as f:
#    f.write(constant_graph.SerializeToString());


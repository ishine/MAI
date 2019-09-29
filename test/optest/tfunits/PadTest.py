import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

#1 2 2 3
input=tf.constant([[[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])
op = tf.pad(input, [3,1,2,0], name='transpose_self')
target=op.eval();
print(target)
#constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['transpose_self'])
#
#with tf.gfile.FastGFile("transpose.pb", mode='wb') as f:
#    f.write(constant_graph.SerializeToString());


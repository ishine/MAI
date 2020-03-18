import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

op = tf.stop_gradient([-1, 1, 0, 1], name="stop_grapdient")

target=op.eval();

print(target)
#constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['range'])
#
#with tf.gfile.FastGFile("range.pb", mode='wb') as f:
#    f.write(constant_graph.SerializeToString());


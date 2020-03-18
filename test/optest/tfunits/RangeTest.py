import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

op = tf.range([3, 4], [18, 10], [5, 3], name="range")

target=op.eval();

print(target)
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['range'])

with tf.gfile.FastGFile("range.pb", mode='wb') as f:
    f.write(constant_graph.SerializeToString());


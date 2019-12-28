import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

#1 2 2 3
input1=tf.constant([[[[1,2,3], [10,5,6]], [[7,8,9], [10,11,12]]]])
input2=tf.constant([[[[3,2,1], [4,5,6]], [[-1,20,100], [12,11,0]]]])
#input2=tf.constant([[7,8,9], [10,11,1]])
op1 = tf.greater(input1, input2, name='greater')
print(op1.eval())

#op = tf.where(op1, (input1 - input2 ), (input2 -input1), name="where")
op = tf.where(op1, input1, input2, name="where")

target=op.eval();

print(target)
#constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['where'])
#
#with tf.gfile.FastGFile("greater.pb", mode='wb') as f:
#    f.write(constant_graph.SerializeToString());


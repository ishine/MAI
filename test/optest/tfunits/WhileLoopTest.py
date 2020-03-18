#import tensorflow as tf
#from tensorflow.python.framework import graph_util
#
#sess = tf.InteractiveSession()
#
##1 3 2 3
#input=tf.constant([[[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], [[13,14,15],[16,17,18]]]])
#op = tf.gather(input, [0], name='gather_self')
#target=op.eval();
#print(target.shape)
#print(target)
##constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['transpose_self'])
##
##with tf.gfile.FastGFile("transpose.pb", mode='wb') as f:
##    f.write(constant_graph.SerializeToString());
#
##3 2 3
#input=tf.constant([[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], [[13,14,15],[16,17,18]]])
#op = tf.gather(input, [0,0], name='gather_self1')
#target=op.eval();
#print(target.shape)
#print(target)

import tensorflow as tf
from tensorflow.python.framework import graph_util
sess = tf.InteractiveSession()
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)


def cond(a, b, c):
    return a<5


def body(a, b, c):
    a += 1
    b += 1
    c +=1
    return a, b, c # same with [a, b, c]



op = tf.while_loop(cond, body, [a,b,c], name='while_loop')
#op.eval()

#with tf.Session() as sess:
print "start run"
sess.run(tf.global_variables_initializer())
print sess.run([a, b, c]) #[5, 6, 7]

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['while_loop'])

with tf.gfile.FastGFile("while_loop.pb", mode='wb') as f:
    f.write(constant_graph.SerializeToString());

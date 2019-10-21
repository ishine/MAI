import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

t1=tf.constant([[[[1.,2.],[3,4]],[[5,6],[7,8]]]], dtype=tf.float32)
t2=tf.constant([[[[1.]],[[2.]]]], dtype=tf.float32)

target = tf.multiply(t1, t2).eval()

# 1 2 2 2 * 1 2 1 1
print(target)

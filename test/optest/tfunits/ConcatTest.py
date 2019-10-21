import tensorflow as tf
from tensorflow.python.framework import graph_util

sess = tf.InteractiveSession()

t1=tf.constant([1., 2.], dtype=tf.float32)
t2=tf.constant([3., 4.], dtype=tf.float32)

target = tf.concat([t1, t2], axis=0).eval()

print("=============[2],[2] axis=0 ==> [4]=========")
print(target)

#2,2,2,1
#t1=tf.constant([[[[],[]],[[],[]]],[[[],[]],[[],[]]]], dtype=tf.float32)
t1=tf.constant([[[[1.],[2]],[[3],[4]]],[[[5],[6]],[[7],[8]]]], dtype=tf.float32)
#2,2,2,2
t2=tf.constant([[[[1.,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]], dtype=tf.float32)

print("=============[2,2,2,1],[2,2,2,2] axis=3 ==> [2,2,2,3]=========")
target = tf.concat([t1, t2], axis=3).eval()
print(target)

#2,2,1,2
t1=tf.constant([[[[1.,2]],[[3,4]]],[[[5,6]],[[7,8]]]], dtype=tf.float32)
#2,2,2,2
t2=tf.constant([[[[1.,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]], dtype=tf.float32)

print("=============[2,2,1,2],[2,2,2,2] axis=2 ==> [2,2,3,2]=========")
target = tf.concat([t1, t2], axis=2).eval()
print(target)

#2,1,2,2
t1=tf.constant([[[[1.,2],[3,4]]],[[[5,6],[7,8]]]], dtype=tf.float32)
#2,2,2,2
t2=tf.constant([[[[1.,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]], dtype=tf.float32)

print("=============[2,1,2,2],[2,2,2,2] axis=1 ==> [2,3,2,2]=========")
target = tf.concat([t1, t2], axis=1, name='concat_').eval()
print(target)
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['concat_'])

with tf.gfile.FastGFile("concat.pb", mode='wb') as f:
    f.write(constant_graph.SerializeToString());

#1,2,2,2
t1=tf.constant([[[[1.,2],[3,4]],[[5,6],[7,8]]]], dtype=tf.float32)
#2,2,2,2
t2=tf.constant([[[[1.,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]], dtype=tf.float32)

print("=============[2,1,2,2],[2,2,2,2] axis=0 ==> [2,3,2,2]=========")
target = tf.concat([t1, t2], axis=0).eval()
print(target)

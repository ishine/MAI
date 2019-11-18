import tensorflow as tf

sess = tf.InteractiveSession()

t1=tf.constant([2., 1.,3, 4], dtype=tf.float32)
t2=tf.constant(0, dtype=tf.int32)
t1=tf.reshape(t1,[2,2]).eval()

target = tf.argmax(t1, t2).eval()

print target

t1=tf.constant([2., 1.,3, 4], dtype=tf.float32)
t2=tf.constant(1, dtype=tf.int32)
t1=tf.reshape(t1,[2,2]).eval()

target = tf.argmax(t1, t2).eval()

print target


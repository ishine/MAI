import tensorflow as tf

sess = tf.InteractiveSession()

t1=tf.reshape(tf.constant([True, False, True, False, False, True, True, True, False, False, True, False]), [2,2,3])

target = tf.reduce_any(t1,[0]).eval()
print target
target = tf.reduce_any(t1,[-3]).eval()
print target
target = tf.reduce_any(t1,[1]).eval()
print target
target = tf.reduce_any(t1,[2]).eval()
print target
target = tf.reduce_any(t1,[0,1]).eval()
print target
target = tf.reduce_any(t1,[2,1]).eval()

print target


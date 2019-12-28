import tensorflow as tf

sess = tf.InteractiveSession()

t1=tf.reshape(tf.constant([True, False, True, False, False, True, True, True, False, False, True, False]), [2,2,3])

#target = tf.reduce_all(t1,[0]).eval()
#target = tf.reduce_all(t1,[-3]).eval()
#target = tf.reduce_all(t1,[1]).eval()
#target = tf.reduce_all(t1,[2]).eval()
target = tf.reduce_all(t1,[0,1]).eval()
target = tf.reduce_all(t1,[2,1]).eval()

print target


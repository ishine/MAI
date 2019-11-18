import tensorflow as tf

sess = tf.InteractiveSession()

t1=tf.constant([1., 2.,3,4], dtype=tf.float32)
t2=tf.constant([4, 3,2,1.], dtype=tf.float32)
t1=tf.reshape(t1,[2,2]).eval()
t2=tf.reshape(t2,[2,2]).eval()

target = tf.pow(t1, t2).eval()

print target

# broadcast
t1=tf.constant([1., 2.,3,4], dtype=tf.float32)
t2=tf.constant([2.], dtype=tf.float32)
t1=tf.reshape(t1,[2,2]).eval()
target = tf.pow(t1, t2).eval()
print target

#broadcast
t1=tf.constant([1., 2.,3,4], dtype=tf.float32)
t2=tf.constant([2.,3], dtype=tf.float32)
t1=tf.reshape(t1,[2,2]).eval()
target = tf.pow(t1, t2).eval()
print target


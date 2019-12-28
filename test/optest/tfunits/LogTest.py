import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1, 2, 3, 4., 0.5])

target = tf.log(t).eval()

print target


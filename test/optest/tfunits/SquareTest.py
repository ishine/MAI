import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1, 2, -1, 0])

target = tf.square(t).eval()

print target


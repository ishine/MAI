import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1.8, 2.2])

target = tf.cos(t).eval()

print target


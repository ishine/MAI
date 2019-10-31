import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1.8, 2.2], dtype=tf.float32)

target = tf.negative(t).eval()

print target


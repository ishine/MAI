import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([-1.8, 2.2, 3, -4], dtype=tf.float32)

target = tf.abs(t).eval()

print target


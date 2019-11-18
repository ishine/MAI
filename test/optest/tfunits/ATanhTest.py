import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([0.5, -0.5], dtype=tf.float32)

target = tf.atanh(t).eval()

print target


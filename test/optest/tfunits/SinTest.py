import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1.8, 2.2])

target = tf.sin(t).eval()

print target

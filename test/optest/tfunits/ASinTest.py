import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([0.5, -0.5])

target = tf.asin(t).eval()

print target


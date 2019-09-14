import tensorflow as tf

sess = tf.InteractiveSession()

dims=tf.constant([2, 3])
value = tf.constant(9)

target = tf.fill(dims, value).eval()

print("======Fill int32==========")
print target

value = tf.constant(1.1)

target = tf.fill(dims, value).eval()

print("======Fill float==========")
print target


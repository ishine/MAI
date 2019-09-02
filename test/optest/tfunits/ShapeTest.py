import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([[[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])

target=tf.shape(t).eval()

print("[1,2,2,3]----")
print target


t=tf.constant([[[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])

target=tf.shape(t).eval()

print("[2,2,3]----")
print target

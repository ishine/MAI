import tensorflow as tf

sess = tf.InteractiveSession()

#[1,2,2,3]
t=tf.constant([[[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])

target=tf.reshape(t, [1, 3, 2, 3]).eval()

print("[1,2,2,3]----")
print target

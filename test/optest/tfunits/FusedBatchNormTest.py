import tensorflow as tf

sess = tf.InteractiveSession()

# 1 2 2 3
t=tf.constant([[[[1.,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])

mean=tf.constant([0.5,0.4,0.3])
scale=tf.constant([0.5,0.4,0.3])
variance=tf.constant([0.3,0.4,0.5])
offset=tf.constant([0.1,0.2,0.3])

target=tf.nn.fused_batch_norm(t, scale, offset, mean, variance, 0.001, 'NHWC', False)[0].eval()

print("NHWC----")
print target

# 1 2 2 3
t=tf.constant([[[[1.,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])

mean=tf.constant([0.5,0.4,0.3])
scale=tf.constant([0.5,0.4,0.3])
variance=tf.constant([0.3,0.4,0.5])
offset=tf.constant([0.1,0.2,0.3])

target=tf.nn.fused_batch_norm(t, scale, offset, mean, variance, 0.002, 'NHWC', False)[0].eval()

print("NHWC epsilon=0.002----")
print target

# Tensorflow CPU support NHWC now
# 1 3 2 2
#t=tf.constant([[[[1.,4], [7,10]], [[2,5],[8,11]],[[3,6],[9,12]]]])
#print(t)
#
#mean=tf.constant([0.5,0.4,0.3])
#scale=tf.constant([0.5,0.4,0.3])
#variance=tf.constant([0.3,0.4,0.5])
#offset=tf.constant([0.1,0.2,0.3])
#
#target=tf.nn.fused_batch_norm(t, scale, offset, mean, variance, 0.001, 'NCHW', False)[0].eval()
#print("NCHW----")
#print target


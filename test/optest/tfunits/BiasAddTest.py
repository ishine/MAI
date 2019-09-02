import tensorflow as tf

sess = tf.InteractiveSession()

#NHWC 1 2 2 3
t=tf.constant([[[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]])

bias=tf.constant([1, 2, 3])

target=tf.nn.bias_add(t, bias).eval()

print("NHWC========================")
print target

#NCHW 1 3 2 2
t=tf.constant([[[[1,4],[7,10]],[[2,5],[8,11]],[[3,6],[9,12]]]])
print(t)

bias=tf.constant([1, 2, 3])

target=tf.nn.bias_add(t, bias, "NCHW").eval()

print("NHWC========================")
print target

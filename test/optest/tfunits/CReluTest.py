import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1.,2.,3.,4.])

target=tf.nn.crelu(t).eval()

print(target)


t=tf.constant([-1.,2.,3.,4., 5,6,7,8])
t=tf.reshape(t,[2, 2,2]).eval()

target=tf.nn.crelu(t).eval()

print(target)
print(target.shape)

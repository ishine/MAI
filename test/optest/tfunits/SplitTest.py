import tensorflow as tf

#help(tf.expand_dims)
sess = tf.InteractiveSession()

t = tf.constant([[1,2,3,4,5,6,7,8,9,10], [11,12,13,14,15,16,17,18,19,20]])

target=tf.split(t, [4, 5, 1], 1)
print "[2, 10] split with [4, 5, 1] axis is 1-->" + str(target[0].shape)
print "[2, 10] split with [4, 5, 1] axis is 1-->" + str(target[1].shape)
print "[2, 10] split with [4, 5, 1] axis is 1-->" + str(target[2].shape)

print
target=tf.split(t, 2, 1)
t0 = target[0].eval()
t1 = target[1].eval()
print "[2, 10] split into 2 tensor axis is 1-->" + str(target[0].shape)
print "[2, 10] split into 2 tensor axis is 1-->" + str(t0)
print "[2, 10] split into 2 tensor axis is 1-->" + str(target[1].shape)
print "[2, 10] split into 2 tensor axis is 1-->" + str(t1)

print
target=tf.split(t, 2)
t0 = target[0].eval()
t1 = target[1].eval()
print "[2, 10] split into 2 tensor axis is default-->" + str(target[0].shape)
print "[2, 10] split into 2 tensor axis is default-->" + str(t0)
print "[2, 10] split into 2 tensor axis is default-->" + str(target[1].shape)
print "[2, 10] split into 2 tensor axis is default-->" + str(t1)

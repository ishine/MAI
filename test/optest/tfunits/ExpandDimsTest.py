import tensorflow as tf

#help(tf.expand_dims)
sess = tf.InteractiveSession()

t=tf.constant([1,2]) # with shape [2]
target=tf.expand_dims(t, 0)

print "[2] 0 -->" + str(target.shape)

target=tf.expand_dims(t, 1)
print "[2] 1 -->" + str(target.shape)

target=tf.expand_dims(t, -1)
print "[2] -1 -->" + str(target.shape)

target=tf.expand_dims(t, -2)
print "[2] -2 -->" + str(target.shape)

t = tf.zeros([2, 3, 5])

target=tf.expand_dims(t, 0)
print "[2, 3, 5] 0 -->" + str(target.shape)

target=tf.expand_dims(t, 1)
print "[2, 3, 5] 1 -->" + str(target.shape)

target=tf.expand_dims(t, 2)
print "[2, 3, 5] 2 -->" + str(target.shape)

target=tf.expand_dims(t, 3)
print "[2, 3, 5] 3 -->" + str(target.shape)

target=tf.expand_dims(t, -1)
print "[2, 3, 5] -1 -->" + str(target.shape)

target=tf.expand_dims(t, -2)
print "[2, 3, 5] -2 -->" + str(target.shape)

target=tf.expand_dims(t, -3)
print "[2, 3, 5] -3 -->" + str(target.shape)

target=tf.expand_dims(t, -4)
print "[2, 3, 5] -4 -->" + str(target.shape)

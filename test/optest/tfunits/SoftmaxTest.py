import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([1.,2.,3.,4.])

target=tf.nn.softmax(t).eval()

print("softmax dim=default")
print target

t=tf.constant([1.,2.,3.,4.])
target=tf.nn.softmax(t, 0).eval()

print("softmax dim=0")
print target

t=tf.constant([[1.,2.,3.,4.],[5.,6.,7.,8.]])
target=tf.nn.softmax(t).eval()

print("softmax dim=default")
print target

t=tf.constant([[1.,2.,3.,4.],[5.,6.,7.,8.]])
target=tf.nn.softmax(t, 0).eval()

print("softmax dim=0")
print target

t=tf.constant([[1.,2.,3.,4.],[5.,6.,7.,8.]])
target=tf.nn.softmax(t, 1).eval()

print("softmax dim=1")
print target

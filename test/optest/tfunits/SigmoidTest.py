import tensorflow as tf

sess = tf.InteractiveSession()

t=tf.constant([-9,-5.5,-3,-0.5,0,1.25,3,7,8])

target=tf.sigmoid(t).eval()

print target


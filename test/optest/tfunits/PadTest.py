import tensorflow as tf

sess = tf.InteractiveSession()

#pad channel
t=tf.constant([[[[1,2], [3,4]], [[5,6], [7, 8]]]])
paddings=tf.constant([[0,0],[0,0],[0,0],[1,2]])

target=tf.pad(t, paddings).eval()

print("pad channel========================")
print target

#pad batch
t=tf.constant([[[[1],[2],[3]], [[4],[5],[6]], [[7],[8],[9]]]])
paddings=tf.constant([[1,1],[0,0],[0,0],[0,0]])

target=tf.pad(t, paddings).eval()

print("pad batch========================")
print target

#pad all
t=tf.constant([[[[1],[2]],[[3],[4]]]])
paddings=tf.constant([[1,1],[1,1],[1,1],[1,1]])

target=tf.pad(t, paddings).eval()

print("pad batch========================")
print target

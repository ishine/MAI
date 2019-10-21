import tensorflow as tf

sess = tf.InteractiveSession()

t1=tf.constant([83, 66, 75, 255, 240, 249, 64, 29, 25,
46, 29, 38, 143, 126, 135, 62, 27, 23,
216, 208, 191, 114, 105, 89, 209, 183, 153], dtype=tf.float32)
t2=tf.constant([4, 4], dtype=tf.int32)
t1=tf.reshape(t1, [1,3,3,3]).eval()

# tf1.2.1 use stack instead of pack
target = tf.image.resize_bilinear(t1, t2).eval()

print("=============[2],[2] axis=0 ==> [4]=========")
print(target)




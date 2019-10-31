import tensorflow as tf

sess = tf.InteractiveSession()

t1=tf.constant([1., 2.], dtype=tf.float32)
t2=tf.constant([3., 4.], dtype=tf.float32)
print(t1.shape);

# tf1.2.1 use stack instead of pack
target = tf.stack([t1, t2], axis=1).eval()

print("=============[2],[2] axis=0 ==> [4]=========")
print(target)



t1=tf.constant([1., 2.,3,4,5,6,7,8], dtype=tf.float32)
t2=tf.constant([9., 10,11,12,13,14,15,16], dtype=tf.float32)

t1=tf.reshape(t1, [2,1,2,2]).eval()
t2=tf.reshape(t2, [2,1,2,2]).eval()

target = tf.stack([t1, t2], axis=0).eval()

print("=============[2,1,2,2],[2,1,2,2] axis=0 ==> [2,2,1,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=1).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=1 ==> [2,2,1,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=2).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=2 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=3).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=3 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=4).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=4 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=-1).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=-1 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=-2).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=-2 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=-3).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=-3 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

target = tf.stack([t1, t2], axis=-4).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=-4 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)
target = tf.stack([t1, t2], axis=-5).eval()
print("=============[2,1,2,2],[2,1,2,2] axis=-5 ==> [2,1,2,2,2]=========")
print(target)
print(target.shape)

import tensorflow as tf

sess = tf.InteractiveSession()

print("-------------------With1Channels VALID------------")

#NHWC 1,4,4,1
inputData=tf.constant([1.,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
inputData=tf.reshape(inputData,[1,4,4,1]).eval()

#HWIO 3,3,1,1
#filter=tf.constant([[[[1.],[2]],[[3],[-4]]],[[[-1],[1]],[[-1],[1]]],[[[-1],[-1]],[[1],[1]]]])
filter=tf.constant([1.,2,3,4,5,6,7,8,9])
filter=tf.reshape(filter,[3,3,1,1]).eval()

outputShape=tf.constant([1,4,4,1])

#target 2,1,3,3
target=tf.nn.conv2d_transpose(inputData, filter, outputShape, [1,1,1,1], 'SAME').eval()

print(target)


import tensorflow as tf

sess = tf.InteractiveSession()

print("-------------------With1Channels------------")

#NHWC 2,2,4,1
inputData=tf.constant([[[[1.],[2],[3],[4]],[[5],[6],[7],[8]]],[[[9],[10],[11],[12]],[[13],[14],[15],[16]]]])

#OHWI 3,2,2,1
filter1=tf.constant([[[[1.],[2]],[[3],[-4]]],[[[-1],[1]],[[-1],[1]]],[[[-1],[-1]],[[1],[1]]]])
#transpose to HWIO 2,2,1,3
filter=tf.transpose(filter1, [1,2,3,0])
bias=tf.constant([1.,2,3])

#target 2,1,2,3
target=tf.nn.conv2d(inputData, filter, [1,2,2,1], 'VALID').eval()

print target


print("-------------------WithMultiChannels------------")

print("-------------------Pointwise------------")

# -*- encoding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets('./data/demo1_mnist/mnist_data/', one_hot=True)

import random
import numpy as np
import matplotlib.pyplot as plt
import datetime

# 用于清除默认图形堆栈并重置全局默认图形.默认图形是当前线程的一个属性。
# 该tf.reset_default_graph函数只适用于当前线程。
# 当一个tf.Session或者tf.InteractiveSession激活时调用这个函数会导致未定义的行为。
# 调用此函数后使用任何以前创建的tf.Operation或tf.Tensor对象将导致未定义的行为。
tf.reset_default_graph()
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, shape=[None, 28,28,1])  #shape in CNNs is always the x height x width x color channels
y_ = tf.placeholder(tf.float32, shape=[None, 10]) #shape is always the x number of classes

#卷积层改变特征的个数，池化层只会改变大小。
def conv2d(x, W): # 一次卷积一次映射（relu)
    # strides是步长　batch_size==1 , h滑多少, w滑多少, channel(feature_map)==1,  paddings=same 补充0
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')
def max_pool__2x2(x):
    # kernel_size : batch_size==1, h, w, feature_map(channel)==1, strides:因为h,w == 2 所以 strides必须1,2,2,1
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# truncated_normal()  函数：
# 从截断的正态分布中取随机值。生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值两个标准差的值则丢弃重新选择。
 # kernel=5*5 1是img的channel（1是黑白，3是彩色） 32 kinds of kernel(32不同的核，会产出32个特征值）
W_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # shape就是W卷积想得到多少特征值
h_conv1 = conv2d(x, W_conv1) + b_conv1  # 执行卷积的操作。
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool__2x2(h_conv1)

#Second Conv and Pool Layers
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1)) # 32：和输入上挂钩的， 64： 和输出上挂钩的
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))  # 输出64个特征值
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)) + b_conv2
h_pool2 = max_pool__2x2(h_conv2)  # 变小了

#First Fully Connected Layer
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1)) #input: 28*28*1 ， 1次pool：14*14*feature   2次pool:7*7*feature
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) #看 特征 1024

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   #执行fc1
# 为了执行fc1： 已知参数 h_pool2,            需要输出  X * 1024
#      tf.reshape(h_pool2, [-1,7*7*64])   *     tf.truncated_normal([7*7*64, 1024]）

# summary:
# conv1 =    conv2d(x, W_conv1) + b_conv1    # 需要定义 w_conv1, b_conv1
# conv1 =    relu(conv1)
# pool1 =    max_pool__2x2(conv1)
# conv2 =    conv2d(pool1, W_conv2) + b_conv2    # 需要定义 w_conv2, b_conv2
# conv2 =    relu(conv2)
# pool2 =    max_pool__2x2(conv2)
# fc1   =    tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # 需要定义w_fc1, b_fc1, 根据w_fc1   h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
# dropout  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# fc2   =    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   #需要定义w_fc2, b_fc2


#Dropout Layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Second Fully Connected Layer
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

#Final Layer
y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
trainStep = tf.train.AdamOptimizer().minimize(crossEntropyLoss)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

batch_size = 50
with tf.device('/gpu:1'):
    for i in range(1000):
        batch = mnist.train.next_batch(batch_size)
        trainInputs = batch[0].reshape([batch_size, 28,28,1]) #
        trainLabels = batch[1]

        if i%100 == 0:
            trainAccuracy = accuracy.eval(session=sess, feed_dict={x:trainInputs, y_:trainLabels, keep_prob:1.0})  #过拟合
            print("step %d, training accuracy %g" % (i, trainAccuracy))

        trainStep.run(session=sess, feed_dict={x: trainInputs, y_: trainLabels, keep_prob: 0.5}) # dropout 50%
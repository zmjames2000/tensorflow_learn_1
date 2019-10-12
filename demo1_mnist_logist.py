# -*- encoding:utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
mnist = input_data.read_data_sets('./data/demo1_mnist/mnist_data/', one_hot=True)


numClasses = 10
inputSize = 784
numHiddenUnits = 50
trainingIterations = 10000
batchSize = 100

X = tf.placeholder(tf.float32, shape=[None, inputSize])
y = tf.placeholder(tf.float32, shape=[None, numClasses])

#参数初始化
W1 = tf.Variable(tf.truncated_normal([inputSize, numHiddenUnits], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1), [numHiddenUnits])
W2 = tf.Variable(tf.truncated_normal([numHiddenUnits, numClasses], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1), [numClasses])

#网络结构
hiddenLayerOutput = tf.matmul(X, W1)+B1
hiddenLayerOutput = tf.nn.relu(hiddenLayerOutput)
finalOutput = tf.matmul(hiddenLayerOutput, W2)+B2
finalOutput = tf.nn.relu(finalOutput)

#网络迭代
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=finalOutput))
opt = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

correct_prediction = tf.equal(tf.argmax(finalOutput, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

with tf.device('/gpu:1'):
    for i in range(trainingIterations):
        batch = mnist.train.next_batch(batchSize)
        batchInput = batch[0]
        batchLabel = batch[1]
        _, trainingLoss = sess.run([opt, loss], feed_dict={X:batchInput, y:batchLabel})
        if i%1000 == 0:
            trainAccuracy = accuracy.eval(session=sess, feed_dict={X:batchInput, y:batchLabel})
            print("step %d, training accuracy %g" % (i, trainAccuracy))
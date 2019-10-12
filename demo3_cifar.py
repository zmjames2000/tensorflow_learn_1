# -*- encoding:utf-8 -*-

import numpy as np
import os
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# mean(axis)函数：求平均值。对m*n的矩阵来说
# axis=0:压缩行，对各列求平均值，返回1*n矩阵。
# axis=1:压缩列，对各行求平均值，返回m*1矩阵。
# axis不设置值，对m*n个数求平均值，返回一个实数。

# reshape()函数：改变数组的形状。
# reshape((2,4)):变为一个二维数组；reshape((2,2,2)):变为一个三维数组
# 当有一个参数为-1时，会根据另一个参数的维度计算数组的另外一个shape属性值。
# 如reshape(data.shape[0],-1):行为data.shape[0]行，列自动算出。data.shape[0]:data第一维的长度。

DIR_PATH = r'./data/demo3_cifar/cifar-10-batches-py'

def clean(data):
    imgs = data.reshape(data.shape[0], 3, 32, 32)  # data.shape[0]= batch_size
    grayscale_imgs = imgs.mean(1)
    cropped_imgs = grayscale_imgs[:,4:28, 4:28]
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    img_size = np.shape(img_data)[1]
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1.0/np.sqrt(img_size))
    normalized = (img_data - meansT) / adj_stds
    return  normalized

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

# hstack(a,b,c,d):水平把数组堆叠起来
# vstack(a,b,c,d):竖直把数组堆叠起来
def read_data(dir):
    names =  unpickle('{}/batches.meta'.format(dir))['label_names']
    # print('names:', names)
    data, labels = [], []
    for i in range(1,6):
        filename = '{}/data_batch_{}'.format(dir, i)
        batch_data = unpickle(filename)
        if len(data) > 0:
            data = np.vstack((data, batch_data['data']))
            labels = np.hstack((labels, batch_data['labels']))
        else:
            data = batch_data['data']
            labels = batch_data['labels']
    # print(np.shape(data), np.shape(labels))

    data = clean(data)
    data = data.astype(np.float32)
    return names, data, labels

# show img_data
random.seed(1)

names,data,labels = read_data(DIR_PATH)

# random.sample(sequence,k)函数
#  从指定序列中随机获取指定长度的片段。
# plt.subplot(r,c,num)函数
# 当需要包含多个子图时使用，分成r行和c列，从左到右从上到下对每个子区进行编号，num指定创建的对象在哪个区域。

def shoe_some_examples(names, data, labels):
    plt.figure()
    rows, cols = 4, 4
    random_idxs = random.sample(range(len(data)), rows*cols)
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        j = random_idxs[i]
        plt.title(names[labels[j]])
        img = np.reshape(data[j,:],(24,24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('./data/demo3_cifar/cifar_examples.png')

shoe_some_examples(names, data, labels)

# raw_data = data[4, :]
# # raw_img = np.reshape(raw_data, (24,24))
# # plt.figure()
# # plt.imshow(raw_img, cmap='Greys_r')
# # plt.show()


def show_conv_results(data, filename=None):
    plt.figure()
    rows, clos = 4, 8
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        plt.subplot(rows, clos, i+1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(os.path.join(DIR_PATH, filename))
    else:
        plt.show()

def show_weights(W, filename=None):
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(W)[3]):
        img = W[:,:,0,i]
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(os.path.join(DIR_PATH,filename))
    else:
        plt.show()


#=======================================================================================================
# def conv2d(x, W):
#     return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')
#
# def max_pool_2x2(x,k):
#     return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')
#
# raw_data = data[4, :]
# x = tf.reshape(raw_data, shape=[-1,24,24,1])
# W = tf.Variable(tf.random_normal([5,5,1,32]))
# b = tf.Variable(tf.random_normal([32]))
#
# # conv1 = tf.nn.relu(conv2d(x, W) + b)
# conv = conv2d(x, W)
# conv_with_b = tf.nn.bias_add(conv, b)
# conv_out = tf.nn.relu(conv_with_b)
# h_pool = max_pool_2x2(conv_out, k=2)
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# 用来显示权重等信息
# with tf.device('/gpu:1'):
#     W_val = sess.run(W)
#     print('Weights:')
#     show_weights(W_val)
#
#     conv_val = sess.run(conv)
#     print('convolution resluts:')
#     print(np.shape(conv_val))
#     show_conv_results(conv_val)
#
#     conv_out_val = sess.run(conv_out)
#     print('convolution with bias and relu:')
#     print(np.shape(conv_out_val))
#     show_conv_results(conv_out_val)
#
#     maxpool_val = sess.run(h_pool)
#     print('maxpool after all the convolutions:')
#     print(np.shape(maxpool_val))
#     show_conv_results(maxpool_val)
#=======================================================================================================

x = tf.placeholder(tf.float32, [None, 24*24])
y = tf.placeholder(tf.float32, [None, len(names)])
W1 = tf.Variable(tf.random_normal([5,5,1,64]))
b1 = tf.Variable(tf.random_normal([64]))
W2 = tf.Variable(tf.random_normal([5,5,64, 64]))
b2 = tf.Variable(tf.random_normal([64]))
W3 = tf.Variable(tf.random_normal([6*6*64, 1024]))
b3 = tf.Variable(tf.random_normal([1024]))
W_out = tf.Variable(tf.random_normal([1024, len(names)]))
b_out = tf.Variable(tf.random_normal([len(names)]))

def conv_layer(x, W, b):
    conv = tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')
    conv_with_b = tf.nn.bias_add(conv, b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out

def maxpool_layer(conv, k=2):
    return tf.nn.max_pool(conv, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def norm(x):
    return tf.nn.lrn(x, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)

def model():
    x_reshaped = tf.reshape(x, shape=[-1,24,24,1])
    conv_out1 = conv_layer(x_reshaped, W1, b1)
    norm1 = norm(conv_out1)
    maxpool_out1 = maxpool_layer(norm1)
    # 提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。
    # 推荐阅读http://blog.csdn.net/banana1006034246/article/details/75204013
    conv_out2 = conv_layer(maxpool_out1, W2, b2)
    norm2 = norm(conv_out2)
    maxpool_out2 = maxpool_layer(norm2)

    maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]]) #[None, 6, 6, 64]   [6*6*64]
    # print(maxpool_reshaped.get_shape().as_list())
    local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
    local_out = tf.nn.relu(local)

    out = tf.add(tf.matmul(local_out, W_out), b_out)
    return  out

learning_rate = 0.001
model_op = model()

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_op, labels=y)
)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 64
with tf.device('/gpu:0'):
    onehot_labels = tf.one_hot(labels, len(names), axis=-1)  # 把值转化成概率
    onehot_vals = sess.run(onehot_labels)

    for j in range(0, 1000):
        avg_accuracy_val = 0.
        batch_count = 0.
        for i in range(0, len(data), batch_size):
            batch_datas = data[i:i+batch_size, :]
            batch_labels = onehot_vals[i:i+batch_size,:]
            _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x:batch_datas, y:batch_labels})
            avg_accuracy_val += accuracy_val
            batch_count += 1.

        avg_accuracy_val /= batch_count
        print('Epoch {}. Avg accuracy {}'.format(j, avg_accuracy_val))

while True: input('>>> press any key continue...')



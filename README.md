# tensorflow_learn_1
learn how to use tensorflow
[https://blog.csdn.net/zmjames2000/article/category/9351992](https://blog.csdn.net/zmjames2000/article/category/9351992)



==================================================

## demo1_mnist_logist.py
简单的用tensorflow 搭建简单的网络结构

## demo2_CNN.py	
简单的用tensorflow 搭建CNN网络结构

summary:<br>
conv1 =    conv2d(x, W_conv1) + b_conv1    # 需要定义 w_conv1, b_conv1<br>
conv1 =    relu(conv1)<br>
pool1 =    max_pool__2x2(conv1)<br>
conv2 =    conv2d(pool1, W_conv2) + b_conv2    # 需要定义 w_conv2, b_conv2<br>
conv2 =    relu(conv2)<br>
pool2 =    max_pool__2x2(conv2)<br>
fc1   =    tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   # 需要定义w_fc1, b_fc1, 根据w_fc1   h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])<br>
dropout  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)<br>
fc2   =    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   #需要定义w_fc2, b_fc2<br>

## demo3_cifar.py  可以学习以下如何定义CNN和使用CNN
简单的使用 tensorflow 训练cifar10数据集<br>
def show_conv_results(data, filename=None):<br>
def show_weights(W, filename=None):<br>


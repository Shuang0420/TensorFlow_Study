#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 加载 MNIST 数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 构建 Softmax 回归模型
# x 是一个占位符placeholder，在TensorFlow运行计算时输入这个值。我们希望能够输入任意数量的MNIST图像，每一张图展平成784维的向量。我们用2维的浮点数张量来表示这些图，这个张量的形状是[None，784 ]。（这里的None表示此张量的第一个维度可以是任何长度的。）
x = tf.placeholder(tf.float32, [None, 784])

# 权重值
W = tf.Variable(tf.zeros([784,10]))

# 偏离值
b = tf.Variable(tf.zeros([10]))

# 类别预测 － softmax 模型
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 构建代价函数
# 正确值
y_ = tf.placeholder("float", [None,10])

# 损失函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初始化变量
init = tf.initialize_all_variables()

# 在session里启动模型
sess = tf.Session()
sess.run(init)

# 开始训练模型，这里我们让模型循环训练1000次！
for i in range(1000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})


# 评估模型
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy,feed_dict={x: mnist.test.images, y_: mnist.test.labels})

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:10:09 2019

@author: lwyan
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#加载mnist_inference中定义的常量和前行传播的函数
import mnist_inference
import os
 
#配置神经网络的参数
BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.9
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
 
#模型保存的文件名和路径
MODEL_SAVE_PATH="D:/Program Files/PythonProject/tensorflow/chapter2/model"
MODEL_NAME="mymodel.ckpt"
 
 
def train(mnist):
	#定义输入输出placeholder()
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
 
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    #将代表轮数的变量指定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
 
    #定义损失函数、学习率、滑动平均操作以及训练过程
 
    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    #在所有代表神经网络的变量上使用滑动平均。tf.trainable_variables返回的是需要训练的变量列表
    #即GRAPHKets.TRAINABLE_VARIABLES中的元素，即没有指定trainable=False的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #计算交叉熵，分类问题只有一个答案时使用sparse_softmax_cross_entropy_with_logits函数
    #para1：神经网络前向传播的结果，para2：训练数据的正确答案
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
 
    #计算在当前batch中所有交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        #指定指数衰减法的学习率
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    #优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #定义上下文的依赖关系
	#只有在 variables_averages_op被执行以后，上下文管理器中的操作才会被执行
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')
 
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
 
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
            	#每训练1000轮输出一次损失函数的大小
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                #保存当前的模型。这里给出了global_step的参数，这样可以让每一个被保存的模型
                #文件名的末尾加上训练的轮数
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
 
def main(argv=None):
    mnist = input_data.read_data_sets("D:/Program Files/PythonProject/tensorflow/mnist_data/", one_hot=True)
    train(mnist)
 
if __name__ == '__main__':
#    tf.app.run()
    main()
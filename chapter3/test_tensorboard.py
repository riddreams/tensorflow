# -*- coding: utf-8 -*-

import tensorflow as tf

input1 = tf.constant([1.0,2.0,3.0],name="input1")
input2 = tf.Variable(tf.random_uniform([3]),name="input2")
output = tf.add_n([input1,input2],name="add")

sess = tf.Session()

# cd D:/Program Files/PythonProject/tensorflow/chapter3
# tensorboard --logdir=log
writer = tf.summary.FileWriter("D:/Program Files/PythonProject/tensorflow/chapter3/log",sess.graph)
writer.close()
sess.close()
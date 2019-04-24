# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.client import device_lib

# 输出可用CPU和GPU信息
print(device_lib.list_local_devices())

with tf.device('/cpu:0'):
    a = tf.constant([1.0,2.0,3.0],shape=[3],name='a')
    b = tf.constant([1.0,2.0,3.0],shape=[3],name='b')
with tf.device('/gpu:1'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
sess.close()

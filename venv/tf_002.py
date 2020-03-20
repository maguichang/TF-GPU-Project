# -*- coding:utf-8 -*-
# @Time :2020/3/20 11:42
# Author :Ma Gui chang
# Email: mgc5320@163.com

import tensorflow as tf

# 定义一个随机数
random_float = tf.random.uniform(shape=())

# 定义一个有2个元素的零向量
zero_vector = tf.zeros(shape=(2), dtype = tf.int32)

# 定义两个2*2的常量矩阵
A = tf.constant([[1.,2.],[3.,4.]])
B = tf.constant([[5.,6.],[7.,8.], ])
print(random_float)
print(A.shape)      # 输出(2, 2)，即矩阵的长和宽均为2
print(A.dtype)      # 输出<dtype: 'float32'>
print(A.numpy())    # 输出[[1. 2.]
                    #      [3. 4.]]
C = tf.add(A, B)    # 计算矩阵A和B的和
D = tf.matmul(A, B) # 计算矩阵A和B的乘积
print(C)
print(D)
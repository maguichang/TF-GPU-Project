# -*- coding:utf-8 -*-
# @Time :2020/3/20 13:14
# Author :Ma Gui chang
# Email: mgc5320@163.com

"""
tf 自动求导机制
"""
import tensorflow as tf

x = tf.Variable(initial_value=3.)
"""
求导记录器
在tf.GradientTape()的上下文内，所有计算步骤都会被记录以用于求导
"""
# with tf.GradientTape() as tape:
#     y = tf.square(x)
# # 计算y关于x的导数
# y_grad = tape.gradient(y, x)
# print([y, y_grad])

X = tf.constant([[1.,2.],[3.,4.]])
y = tf.constant([[1.],[2.]])
w = tf.Variable(initial_value = [[1.],[2.]])
b = tf.Variable(initial_value = 1.)
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X,w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b])
print(L.numpy(),w_grad.numpy(), b_grad.numpy())

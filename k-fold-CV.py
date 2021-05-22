# -*- coding: utf-8 -*-
# by Kisna 21.5.22

'''K-Fold Cross Validation'''

import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf # tf版本限定
tf.disable_v2_behavior()
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import StratifiedKFold
from hk_function import cross_validation

k = 20 # 5，10

# 图片和标签打乱函数
def shuffer_images_and_labels(images, labels):
    shuffle_indices = np.random.permutation(np.arange(len(images)))
    shuffled_images = images[shuffle_indices]
    shuffled_labels = labels[shuffle_indices]
    return shuffled_images, shuffled_labels

def batch_iter(images,labels, batch_size, epoch_num, shuffle=True):
    
    data_size = len(images)
    
    num_batches_per_epoch = int(data_size / batch_size)  # 样本数/batch块大小,多出来的“尾数”，不要了
    
    for epoch in range(epoch_num):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            
            shuffled_data_feature = images[shuffle_indices]
            shuffled_data_label   = labels[shuffle_indices]
        else:
            shuffled_data_feature = images
            shuffled_data_label = labels

        for batch_num in range(num_batches_per_epoch):   # batch_num取值0到num_batches_per_epoch-1
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield(shuffled_data_feature[start_index:end_index] , shuffled_data_label[start_index:end_index])

# 全连接层构建函数
def fcn_layer(inputs,  #输入数据
             input_dim,  #输入神经元数量
             output_dim,  #输出神经元数量
             activation=None):  #激活函数
    W = tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev = 0.1))  #随机数生成的数据如果比标准差两倍还要大
                                                              #就要被替换掉
    b = tf.Variable(tf.zeros([output_dim]))  #初始化为0
    
    XWb = tf.matmul(inputs,W)+b
    
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
        
    return outputs


# 读取数据集
    
mnist = input_data.read_data_sets('./mnist_dataset', one_hot=True)

#把train数据集的标签和图像打乱
total_images = mnist.train.images
total_labels = mnist.train.labels
total_images, total_labels = shuffer_images_and_labels(total_images, total_labels)

#验证集保持不变
total_validation_images = mnist.validation.images
total_validation_labels = mnist.validation.labels
total_validation_images, total_validation_labels = shuffer_images_and_labels(
        total_validation_images, total_validation_labels)

# 简单划分前50000个为训练集，后5000个为测试集
origin_images_train,origin_labels_train,origin_images_test,origin_labels_test = cross_validation(total_images, total_labels, k)


# 构建和训练模型
def train_and_test(images_train, labels_train, images_test,
                   labels_test,images_validation, labels_validation):
    
    #构建输入层
    x = tf.placeholder(tf.float32, [None,784], name="X")

    #0-9 一共10个数字=>10个类别
    y = tf.placeholder(tf.float32, [None, 10], name="Y")
    
    
    #隐藏层
    h1 = fcn_layer(inputs = x,
              input_dim = 784,
              output_dim = 256,
              activation = tf.nn.relu)
    
    #h2 = fcn_layer(inputs=h1,
     #         input_dim=256,
      #        output_dim=64,
       #       activation=tf.nn.relu)
    
    #输出层
    forward = fcn_layer(inputs = h1,
                   input_dim = 256,
                   output_dim = 10,
                   activation = None)
    pred = tf.nn.softmax(forward)
    
    # 交叉熵损失函数(概率分布距离的刻画)
    loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = forward,labels = y))
    
    # 超参数设置
    train_epochs = 40 # 训练轮数
    batch_size = 50   # 单次训练样本数（批次大小）
    total_batch = int(mnist.train.num_examples / batch_size)  # 一轮训练有多少批次
    display_step = 1  # 显示粒度
    learning_rate = 0.001  # 学习率
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)
    
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))

    # 准确率计算，将布尔值转化为浮点数，并计算平均值
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    #save_step = 5
    
    #saver = tf.train.Saver()
    
    from time import time
    startTime = time()
    
    sess = tf.Session() # 声明会话
    init = tf.global_variables_initializer() # 初始化变量
    sess.run(init)
    
    # 迭代训练
    for epoch in range(train_epochs):
        for a,b in batch_iter(images_train,labels_train, batch_size, 1, shuffle=True):
            xs, ys = a, b
            sess.run(optimizer,feed_dict={x:xs,y:ys})
            
        
        loss,acc = sess.run([loss_function,accuracy],
                       feed_dict = {x:images_validation, y:labels_validation})
    
        if(epoch+1) % display_step==0:
            print("train epoch:",'%02d' %(epoch+1),"Loss=","{:.9f}".format(loss),\
             "accuracy=","{:.4f}".format(acc))
            
        #if(epoch+1) % save_step==0:
         #   saver.save(sess,os.path.join(ckpt_dir,"mnist_h256_model_{:06d}.ckpt".format(epoch+1)))
          #  print("mnist_h256_model_{:06d}.ckpt saved".format(epoch+1))
            
    #saver.save(sess,os.path.join(ckpt_dir,"mnist_h256_model.ckpt"))
    #print("Model saved!")
    
    duration=time()-startTime    
    print("train finished takes:","{:.2f}".format(duration))
    
    # 测试集测试
    accu_test = sess.run(accuracy,feed_dict={x:images_test,y:labels_test})
    print("test accuracy:",accu_test)
    return accu_test

accu_test_f = 0
for i in range(k):
    accu_test = train_and_test(origin_images_train, origin_labels_train, origin_images_test,
                   origin_labels_test,total_validation_images, total_validation_labels)
    accu_test_f = accu_test+accu_test_f
    
print("test accuracy:",accu_test_f/k)
    



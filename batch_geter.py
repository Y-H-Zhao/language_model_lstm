# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:01:59 2018

@author: ZYH
"""
import numpy as np
#import tensorflow as tf

train_data="ptb.train" #单词编号数据
train_batch_size=20
train_num_step=35

#定义从文件读取数据，并返回包含单词编号的数组
def read_data(file_path):
    with open(file_path,"r") as f_in:
        #将整个文档都近一个字符串 line.strip()复制line内容为字符串
        id_string=' '.join([line.strip() for line in f_in.readlines()])
    id_list=[int(w) for w in id_string.split()] #转化为整数
    return id_list

#定义取得batch的函数
def make_batches(id_list,batch_size,num_step):
    #计算batch数量，每个batch 有batch_size * num_step个单词
    num_batches=(len(id_list)-1)//(batch_size * num_step)
    
    #将数据整理成 维度为 [batch_size,num_batches* num_step]的二维数据
    data=np.array(id_list[:num_batches*batch_size*num_step])
    data=np.reshape(data,[batch_size,num_batches*num_step]) #每num_step列是一个batch
    #沿着第二个维度切分成num_batches*batch 存入一个数组
    data_batches = np.split(data,num_batches,axis=1)
    
    #重复上述操作，但是每个位置向右移一位，得到RNN每一步输出所需预测的下一个单词
    label = np.array(id_list[1:num_batches*batch_size*num_step+1])
    label=np.reshape(label,[batch_size,num_batches*num_step]) #每num_step列是一个batch
    #沿着第二个维度切分成num_batches*batch 存入一个数组 切开后 每一部分是一个batch 保存为一个大数组
    label_batches = np.split(label,num_batches,axis=1)
    #返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵
    return list(zip(data_batches,label_batches))
def main():
    train_batches=make_batches(read_data(train_data),
                               train_batch_size,train_num_step)
    #训练代码...
    #print(train_batches)
    
if __name__=="__main__":
    main()

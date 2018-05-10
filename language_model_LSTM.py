# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:03:31 2018

@author: ZYH
"""
'''
0.分词 得到句子的单词组成形式（英文不用分词，中文需要分词）
1.单词的频率统计，得到词汇表
2.单词编码 得到分词后的句子的编码形式
3.对编码形式的句子进行训练，验证，测试
'''
import numpy as np 
import tensorflow as tf

train_data="ptb.train" #数据路径
eval_data="ptb.valid"
test_data="ptb.test"

#超参数
hidden_size=300 #隐藏层规模
num_layers=2 #lstm结构层数
vocab_size=10000 #词典规模
train_batch_size=20 #batch_size
train_num_step=35 #sequances_lehgth

eval_batch_size=1 #验证数据的batch_size
eval_num_step=1
num_epoch=5 #训练轮数
lstm_keep_prob=0.9
embedding_keep_prob=0.9 #drop
max_grad_norm=5 #用于控制梯度膨胀的梯度上限
share_emb_and_softmax=True #共享参数

'''
老规矩 先构造数据结构，batch_geter.py中已经对这个程序就行了详细介绍
'''
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


'''
其次 定义模型，定义训练, main()中train,这一套可以根据不同情况切分为几部分
'''
#定义一个ptbmodel类，方便维护网络的状态
class ptbmodel(object):
    def __init__(self,is_training,batch_size,num_steps):
        #记录使用的batch大小和截断长度
        self.batch_size=batch_size
        self.num_steps=num_steps
        
        #定义每一步的输入个预期输出 两者的维度都是[batch_size,num_steps]
        self.input_data=tf.placeholder(tf.int32,[batch_size,num_steps])
        self.targets=tf.placeholder(tf.int32,[batch_size,num_steps])
        
        #定义使用LSTM结构为循环体结构，且使用dropout
        dropout_keep_prob=lstm_keep_prob if is_training else 1.0
        lstm_cells=[
                tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.BasicLSTMCell(hidden_size),
                        output_keep_prob=dropout_keep_prob) 
                for _ in range(num_layers)]
        cell=tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
        
        #初始化最初的状态 全零，只在每个epoch初始化的第一个batch使用
        self.initial_state=cell.zero_state(batch_size,tf.float32)
        
        #第一层 词向量矩阵
        embedding = tf.get_variable("embedding",[vocab_size,hidden_size])
        
        #单词转化为词向量 寻找生成词向量
        inputs=tf.nn.embedding_lookup(embedding,self.input_data)
        
        #训练是使用dropout
        if is_training:
            inputs=tf.nn.dropout(inputs,embedding_keep_prob)
            
        #定义输出列表 将不同时刻的LSTM输出收集起来 再一起提供给softmax
        outputs=[]
        state=self.initial_state #最开始使用初始状态
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps): #训练次数
                if time_step>0: tf.get_variable_scope().reuse_variables()
                cell_output, state=cell(inputs[:,time_step,:],state)
                outputs.append(cell_output)
                
        #把队列展开成[batch_size,hidden*num_steps]的形状
        #然后再reshape成[batch*num_steps,hidden_size]的形状
        output=tf.reshape(tf.concat(outputs,1),[-1,hidden_size])
            
        #softmax层
        if share_emb_and_softmax:
            weight=tf.transpose(embedding)
            
        else:
            weight=tf.get_variable("weight",[hidden_size,vocab_size])
        bias=tf.get_variable("bias",[vocab_size])
        logits=tf.matmul(output,weight)+bias
        
        #定于交叉熵 这部分可以放在训练部分
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets,[-1]),
                logits=logits)
        self.cost=tf.reduce_sum(loss)/batch_size
        self.final_state=state
        
        #训练 反向传播
        if not is_training: return
        
        trainable_variables=tf.trainable_variables()
        #控制梯度大小 定义优化方法和步骤
        grads,_=tf.clip_by_global_norm(
                tf.gradients(self.cost,trainable_variables),max_grad_norm)
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
        self.train_op=optimizer.apply_gradients(
                zip(grads,trainable_variables))
'''
使用定义模型在data上运行train_op，并返回全部数据上的perplexity
'''
def run_opech(session,model,batches,train_op,output_log,step):
    #辅助变量
    total_costs=0
    iters=0
    state=session.run(model.initial_state)
    #训练一个epoch
    for x,y in batches:
        #在当前batch上运行train_op并计算损失值
        cost,state,_=session.run(
                [model.cost,model.final_state,train_op],
                {model.input_data:x,model.targets:y,
                 model.initial_state:state})
        total_costs += cost
        iters += model.num_steps
        
        #训练输出日志
        if output_log and step %100 ==0:
            print("After {0} steps,perplexity is {1}".format(step,
                  np.exp(total_costs/iters)))
            
        step+=1
        
    #返回模型的perplexity
    return step,np.exp(total_costs/iters)
#主函数
def main():
    #定义初始化函数
    initializer=tf.random_uniform_initializer(-0.05,0.05)
    
    #定义训练用的模型
    with tf.variable_scope("language_model",reuse=None,initializer=initializer):
        train_model=ptbmodel(True,train_batch_size,train_num_step)
        
    #定义验证用的模型 他与train_model权值共享，但是没有dropout
    with tf.variable_scope("language_model",reuse=True,
                           initializer=initializer):
        eval_model=ptbmodel(False,eval_batch_size,eval_num_step)
        
    #训练模型
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        train_batches=make_batches(
                read_data(train_data),train_batch_size,train_num_step)
        eval_batches=make_batches(
                read_data(eval_data),eval_batch_size,eval_num_step)
        test_batches=make_batches(
                read_data(test_data),eval_batch_size,eval_num_step)
        step=0
        for i in range(num_epoch):
            print("In ieration: {}".format(i+1))
            step,train_pplex=run_opech(session,train_model,train_batches,
                                       train_model.train_op,True,step)
            print("Epoch: {} Train pplex: {}".format(i+1,train_pplex))
            
            _,eval_pplex=run_opech(session,eval_model,eval_batches,
                                       tf.no_op(),False,0)
            print("Epoch: {} Eval pplex: {}".format(i+1,eval_pplex))
        
        _,test_pplex=run_opech(session,eval_model,test_batches,
                                       tf.no_op(),False,0)
        print("Epoch: {} test pplex: {}".format(i+1,test_pplex))
        
if __name__ =="__main__":
    main()
            
        




    


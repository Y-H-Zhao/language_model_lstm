# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:13:48 2018

@author: ZYH
"""
import codecs #codecs专门用作编码转换 防止乱码
import collections 
from operator import itemgetter

raw_data="ptb.train.txt" #训练数据
vocab_output='ptb.vocab' #输出词汇表文件

counter = collections.Counter() #统计单词频率
with codecs.open(raw_data,"r","utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word]+=1
#按词频对单词排序
sort_word_to_cnt=sorted(counter.items(),
                        key=itemgetter(1),
                        reverse=True)
sorted_words=[x[0] for x in sort_word_to_cnt]

#稍后我们需要在文本换行处加入句子结束符“<eos>”,这里预先将其加入词汇表
sorted_words=["<eos>"]+sorted_words
#在其他问题中 还需要将“<sos>”句子起始符，“<unk>”低频替代符加入
#即：
#sorted_words=["<eos>","<sos>","<unk>"]+sorted_words


with codecs.open(vocab_output,'w','utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word +"\n")

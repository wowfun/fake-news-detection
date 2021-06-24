from django.shortcuts import render

import os

import jieba
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForSequenceClassification
import torch
import torch.optim as optim
from torch import nn
from torch import t
from ERNIE_Model import *

def index(request):
    if request.method == "POST":
        x = request.POST.get("X_test", None)
        pred_result=''
        model_name=request.POST.get("model", None)
        model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'bilstm.h5') # 模型保存的文件地址
        model=load_model(model_path) # 加载模型

        print('* get x:',x)
        print('* get model name:',model_name)
        print("* Predict")

        if model_name=='BiLSTM':
            pred_result=bilstm_predict_result(x,model)
        elif model_name=="ERNIE":
            pred_result=ERNIE_predict_result(x)
       
        return render(request,'index.html',{'x':x, 'pred_result': pred_result})
    return render(request,'index.html',{})


## 以下为各个模型的函数接口

# BiLSTM模型接口（加载运行BiLSTM模型）
def bilstm_predict_result(x, model):
    tokenizer_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),'tokenizer.pickle') # 词向量器地址
    max_num_words=100000
    max_seq_len=128

    # 数据预处理
    x=[" ".join([w for w in list(jieba.cut(x))])] # 分词
    x,_=tokenize(x,mode='load',path=tokenizer_path,max_num_words=max_num_words,max_sequence_len=max_seq_len) # 词向量化

    # 调用模型进行预测，并处理成显示在网页上的结果
    prob=model.predict(x)
    label='真' if prob.argmax()==0 else '假'
    result='BiLSTM模型认为 有%s的概率为 %s'%(prob.max(),label)

    return result

# ERINIE模型接口（加载运行BiLSTM模型）
def ERNIE_predict_result(x):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'ernie-1.0') # 模型保存的文件夹地址
    model_dict = os.path.join(os.path.dirname(os.path.dirname(__file__)),'ERNIE0.pkl') # 模型参数保存的地址
    #model_path = "D:/jupyter_codes/UCAS课程作业/大数据分析/大作业/ernie-1.0"
    #model_dict = './模型保存./ERNIE0.pkl'
    
    #读入预训练模型
    tokenizer_ch = BertTokenizer.from_pretrained(model_path)
    model_a = BertModel.from_pretrained(model_path,output_hidden_states=True)
    
    #读取保存的模型参数
    MAX_LEN = 260
    test_model = BERT_senti(tokenizer_ch,model_a,768,64,10,2)
    test_model.load_state_dict(torch.load(model_dict,map_location=torch.device('cpu')))
    
    # 调用模型进行预测，并处理成显示在网页上的结果
    x = x[:258]
    result = test_model.test_one([x])
    result = torch.exp(result)
    result = torch.max(result / torch.sum(result) , dim = 1)
    label = result.indices.item()
    prob = result.values.item()

    label='真' if label == 0 else '假'
    result = 'ERNIE模型认为 有%s的概率为 %s'%(prob,label)
    
    return result


### 以下是用于模型函数接口的辅助函数，可根据自己模型的需要去定义

# 词向量化
def tokenize(lang, mode='load', path=None, max_num_words=None, max_sequence_len=256):  # mode: create or load
    if mode == 'load':
        with open(path, 'rb') as handle:
            lang_tokenizer = pickle.load(handle)
        print('** Load tokenzier from: ', path)
    else:
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_num_words,
                                                               filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~', lower=True)
        lang_tokenizer.fit_on_texts(lang)
        # saving
        with open(path, 'wb') as handle:
            pickle.dump(lang_tokenizer, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print('** Save tokenizer at: ', path)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=max_sequence_len,
                                                           padding='post', truncating='post')  # NOTE
    print('** Total different words: %s.' % len(lang_tokenizer.word_index))

    return tensor, lang_tokenizer
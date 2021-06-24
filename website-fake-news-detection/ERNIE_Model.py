from transformers import BertTokenizer, BertModel, BertForMaskedLM,BertForSequenceClassification
import torch
import torch.optim as optim
from torch import nn
from torch import t
class BERT_senti(nn.Module):
    
    def __init__(self,tokenizer,model,embedding_dim,lstm_hidden_dim,lin_hidden_dim,target_size,MAX_LEN = 260,drop_out = 0.2):
        super(BERT_senti,self).__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.sen_len = MAX_LEN
        #for p in self.parameters():
        #    p.requires_grad=False
        
        unfreeze_layers = ['layer.8','layer.9','layer.10','layer.11','bert.pooler','out.']
        
        for name ,param in self.model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
        self.bilstm = nn.LSTM(embedding_dim, lstm_hidden_dim,
                              batch_first=True,
                              bidirectional=True,dropout = drop_out)
        #self.pool1 = nn.AvgPool1d()
        self.fc1 = nn.Linear(lstm_hidden_dim * 2,target_size)
        
    #传入句子列表    
    def forward(self,sentence_batch):
        input_tensor_list = []
        mask_list = []
        for sentence in sentence_batch:
            input_ids = torch.tensor(self.tokenizer.encode(sentence)).unsqueeze(0)
            pad_tensor = torch.tensor([[0] * (self.sen_len - len(input_ids[0]))])
            input_tensor_list.append(torch.cat((input_ids,pad_tensor),1).long())
            mask = torch.cat((torch.tensor([[1] * len(input_ids[0])]),pad_tensor),1)
            mask_list.append(mask)
        
        mask_tensor = torch.cat(mask_list,0)
        outputs = torch.cat(input_tensor_list,0)
        all_,pooled,all_hidden_states = self.model(outputs,attention_mask = mask_tensor)
        #取BERT最后一层对整个句子的编码作为LSTM的输入
        #print(torch.tensor(all_hidden_states[-4:]))
        
        lstm_input = (all_hidden_states[-1] + all_hidden_states[-2] + all_hidden_states[-3] + all_hidden_states[-4]) / 4
        #lstm_input = torch.cat([all_hidden_states[-1],all_hidden_states[-2],all_hidden_states[-3],all_hidden_states[-4]],1)
        out,_ = self.bilstm(lstm_input)
        fc1_out = self.fc1(out[:,-1,:])

        return fc1_out
        
    
    #用dataloader提取数据,batch_size在dataloader设定，一个epoch要采样多少样本在sampler设定
    def train_loader(self,data_loader,epoch_num,batch_size,lr_rate = 5e-5,w_decay = 1e-4):   
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(),lr = lr_rate,weight_decay = w_decay)
     
        
        for epoch in range(epoch_num):
            predict_all = []
            labels_all = []
            for i,(batch_sentence,batch_targets) in enumerate(data_loader):
                scores = self.forward(batch_sentence)
                targets = torch.tensor(batch_targets,dtype = torch.long)
                loss = criterion(scores,targets)
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                #nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                predict_all.append(torch.argmax(scores,dim = 1).cpu().detach().numpy().tolist())
                labels_all.append(batch_targets)

            predict_all = np.array(predict_all).flatten()
            labels_all = np.array(labels_all[0]).flatten()

            f1 = f1_score(labels_all, predict_all, average='macro')
            print("f1",f1)
        return predict_all
    
    def test_one(self,sentence):
        with torch.no_grad():
            result =  self.forward(sentence)
        return result
        
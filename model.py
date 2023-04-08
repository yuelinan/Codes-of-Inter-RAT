import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.distributions.bernoulli import Bernoulli
import numpy as np

import datetime

class MaskGRU(nn.Module): 
    def __init__(self, in_dim, out_dim, layers=1, batch_first=True, bidirectional=True, dropoutP=0.2):
        super(MaskGRU, self).__init__()
        self.gru_module = nn.GRU(in_dim, out_dim, layers, batch_first=batch_first, bidirectional=bidirectional)
        self.drop_module = nn.Dropout(dropoutP)

    def forward(self, inputs ):
        self.gru_module.flatten_parameters()
        input, seq_lens = inputs
        mask_in = input.new(input.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask_in[i, :seq_lens[i]] = 1
        mask_in = Variable(mask_in, requires_grad=False)

        input_drop = self.drop_module(input * mask_in)

        H, _ = self.gru_module(input_drop)

        mask = H.new(H.size()).zero_()
        for i in range(seq_lens.size(0)):
            mask[i, :seq_lens[i]] = 1
        mask = Variable(mask, requires_grad=False)
        output = H * mask

        return output



class InterRAT(nn.Module):
    def __init__(self,args, embedding):
        super(InterRAT, self).__init__()
        print('InterRAT')
        self.embs = embedding
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)
        self.alpha_rationle = args.alpha_rationle
        self.num_labels = args.class_num
        self.device = args.device
        self.Intervention_type = args.Intervention_type
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.encoder = MaskGRU(args.embed_dim, args.lstm_hidden_dim//2,args.lstm_hidden_layer)
        self.x_2_prob_z = nn.Linear(args.lstm_hidden_dim*2, 2)
        self.label2label = nn.Linear(args.lstm_hidden_dim, args.lstm_hidden_dim)
        self.abs =  args.abs
        self.extract_encoder = nn.GRU(args.embed_dim, args.lstm_hidden_dim//2,args.lstm_hidden_layer, batch_first=True, bidirectional=True)

        self.re_encoder = MaskGRU(args.embed_dim, args.lstm_hidden_dim//2,args.lstm_hidden_layer)

        self.ChargeClassifier = nn.Linear(args.lstm_hidden_dim*2, 2) # predict the result
        self.lab_embed_layer = self._create_label_embed_layer()
       
        self.Transformation_first = nn.Linear(self.lstm_hidden_dim , self.lstm_hidden_dim )

    def _create_label_embed_layer(self):
        embed_layer = nn.Embedding(self.num_labels, self.lstm_hidden_dim)
        embed_layer.weight.data.normal_(mean=0, std=0.1)
        embed_layer.weight.requires_grad = True
        return embed_layer


    def forward(self, documents,sent_len):
        eps = 1e-8
        embed = self.embs(documents) 
        mask = torch.sign(documents).float()
        
        
        batch_size = embed.size(0)
        en_outputs = self.encoder([embed,sent_len.view(-1)])  
       
        label_0 = self.lab_embed_layer(torch.tensor([0],dtype=torch.long).to(self.device)) 
        label_1 = self.lab_embed_layer(torch.tensor([1],dtype=torch.long).to(self.device))
        

        label_0 = label_0.unsqueeze(0).repeat(batch_size,1,1)
        label_1 = label_1.unsqueeze(0).repeat(batch_size,1,1)
        label = torch.cat([label_0,label_1],dim=1)
        label1 = self.label2label(label)
        S = torch.bmm(en_outputs, label1.transpose(1, 2))
        A = torch.softmax(S,-1)
        
        A_split = torch.split(A, 1, dim=-1)
        class_represention = 0.05 * torch.bmm(A_split[0], label_0) +  0.95 * torch.bmm(A_split[1], label_1)
        first_en_outputs_label = torch.cat([en_outputs,class_represention],dim=-1)

        z_logits = self.x_2_prob_z(first_en_outputs_label) 

        sampled_seq = F.gumbel_softmax(z_logits,hard=True,dim=2)
        sampled_seq = sampled_seq * mask.unsqueeze(2)

        sampled_num = torch.sum(sampled_seq[:,:,1], dim = 1) 
        sampled_num = (sampled_num == 0).to(self.device, dtype=torch.float32)  + sampled_num
        sampled_word = embed * (sampled_seq[:,:,1].unsqueeze(2)) 

        s_w_feature = self.re_encoder([sampled_word,sent_len.view(-1)])
        s_w_feature = torch.sum(s_w_feature, dim = 1)/ sampled_num.unsqueeze(1)



        label_0 = self.lab_embed_layer(torch.tensor([0],dtype=torch.long).to(self.device)) 
        label_1 = self.lab_embed_layer(torch.tensor([1],dtype=torch.long).to(self.device))
        label = torch.cat([label_0,label_1],dim=0)

        S = torch.mm(s_w_feature, label.T)
        A = torch.softmax(S,-1)
       

        A_split = torch.split(A, 1, dim=-1)
       
        class_represention2 = 0.05 * torch.mm(A_split[0], label_0) +  0.95 * torch.mm(A_split[1], label_1)
        second_en_outputs_label = torch.cat([s_w_feature,class_represention2],dim=-1)


        output = self.ChargeClassifier(second_en_outputs_label) 

        mask_number1 = sampled_seq[:,:,1]
        
        infor_loss = (mask_number1.sum(-1) / (mask.sum(1)+eps) ) - self.alpha_rationle
        if self.abs == 1:
            self.infor_loss = torch.abs(infor_loss).mean()
        else:
            self.infor_loss = infor_loss.mean()
        
        regular =  torch.abs(mask_number1[:,1:] - mask_number1[:,:-1]).sum(1) / (sent_len-1+eps)
        self.regular = regular.mean()


        return output , sampled_seq[:,:,1].unsqueeze(-1), 1





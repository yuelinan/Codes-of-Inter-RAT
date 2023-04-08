import logging
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
from sklearn import metrics
import numpy as np
import torch
import os
import torch.nn as nn
from utils import Data_Process,eval_data_types,make_kv_string
from model import *
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, ExponentialLR

parser = argparse.ArgumentParser(description='InterRAT')

parser.add_argument('--aspect', type=int, default=0, help='aspect')
parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate')
parser.add_argument('--re', type=float, default=0.01, help='re')
parser.add_argument('--ib', type=float, default=0.01, help='ib')

parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')

parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--infor_loss', type=float, default=0.2, help='infor_loss')
parser.add_argument('--regular', type=float, default=0.2, help='regular')

parser.add_argument('--embed_dim', type=int, default=200, help='original number of embedding dimension')
parser.add_argument('--lstm_hidden_dim', type=int, default=150, help='number of hidden dimension')
parser.add_argument('--mask_hidden_dim', type=int, default=200, help='number of hidden dimension')
parser.add_argument('--lstm_hidden_layer', type=int, default=1, help='number of hidden layers')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--class_num', type=int, default=2, help='class_num')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--save_name', type=str, default='model_', help='model_name')
parser.add_argument('--is_emb', type=str, default='training', help='is_emb')
parser.add_argument('--logger_name', type=str, default='training', help='is_emb')
parser.add_argument('--Intervention_type', type=str, default='linear', help='Intervention_type')
parser.add_argument('--min_lr', type=float, default=5e-5)
parser.add_argument('--lr_decay', type=float, default=0.97)
parser.add_argument('--abs', type=int, default=1)
parser.add_argument('--times', type=int, default=1)
args = parser.parse_args()

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO,
                    filename='./'+args.model_name.lower()+'_a'+str(args.aspect)+ '_seed_'+str(args.seed)+ '_lr_'+str(args.lr)+'_rationale_'+str(args.alpha_rationle)+'_time_'+str(args.times) ,
                    filemode='w')
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(args.device)
process = Data_Process(args)
# load data
data_all = process.read_data(args.types)

dataloader = DataLoader(data_all, batch_size = args.batch_size, shuffle=True, num_workers=0, drop_last=False)

test_data_all = process.read_data('test')
test_dataloader = DataLoader(test_data_all, batch_size = 1, shuffle=False, num_workers=0, drop_last=False)

dev_data_all = process.read_data('dev')
dev_dataloader = DataLoader(dev_data_all, batch_size = args.batch_size, shuffle=False, num_workers=0, drop_last=False)

if args.is_emb=='training':
    vectors = process.embedding
    vectors.weight.requires_grad = False
else:
    vectors = 1

for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))


def main():
    # load model
    model = eval(args.model_name)(args, vectors)
    model = model.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # set learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=args.lr_decay)

    model.train()

    re = args.re
    global_step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
        for step,batch in enumerate(tqdm(dataloader)):
            
            optimizer.zero_grad()
           
            label,documents,sent_len = process.process_data(batch,types=args.types)
            documents = documents.to(args.device)
            label = label.to(args.device)
            sent_len = sent_len.to(args.device)
          
            output,_ ,_= model(documents,sent_len)
            
            # caluate loss 
            model_loss = criterion(output,label)
            
            loss = model_loss + args.infor_loss * model.infor_loss + args.regular * model.regular
            
            loss.backward()
            optimizer.step()
        
       
        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        if cur_lr > args.min_lr:
            scheduler.step()
        cur_lr = scheduler.optimizer.param_groups[0]["lr"]
        print("#lr", cur_lr)
        scheduler.optimizer.param_groups[0]["lr"] = max(args.min_lr,cur_lr)


        ###############################
        model.eval()
        predictions_charge = []
        true_charge = []
        for step,batch in enumerate(tqdm(dev_dataloader)):
    
            label,documents,sent_len = process.process_data(batch,types='dev')
            documents = documents.to(args.device)
            sent_len = sent_len.to(args.device)
            true_charge.extend(label.numpy())

            with torch.no_grad():
                output,mask_number,_ = model(documents,sent_len)
            pred = output.cpu().argmax(dim=1).numpy()
            predictions_charge.extend(pred)
        dev_eval = {}
 
        class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 = eval_data_types(true_charge,predictions_charge,num_labels=2)

        dev_eval['Acc'] = class_micro_f1
        dev_eval['macro_precision'] = class_macro_precision
        dev_eval['macro_recall'] = class_macro_recall
        dev_eval['macro_f1'] = class_macro_f1
    
        dev_s = make_kv_string(dev_eval)

        logger.info("model dev {}".format(dev_s))
       
        model.eval()
        percents = 0
        predictions_charge = []
        true_charge = []

        precision_all = 0
        recall_all = 0
        f1_all = 0
        correct = 0
        totalp = 0
        totalr = 0
        for step,batch in enumerate(tqdm(test_dataloader)):
    
            label,documents,sent_len,rationale = process.process_data(batch,types='test')
            documents = documents.to(args.device)
            sent_len = sent_len.to(args.device)
            true_charge.extend(label.numpy())

            with torch.no_grad():
                output,mask_number,_ = model(documents,sent_len)
            pred = output.cpu().argmax(dim=1).numpy()
            predictions_charge.extend(pred)
            
            mask_number = mask_number.cpu()
            mask_number = mask_number.squeeze(0).squeeze(-1)
            percent = int(mask_number.sum()) / mask_number.size(0)
            percents+=percent
            rationale_tensor = torch.zeros(mask_number.size())
            for start,end in rationale:
                rationale_tensor[start:end] = 1

            match = 0
        
            for index,data in enumerate(rationale_tensor):
                if data == mask_number[index]:
                    match += data

            should_mathch = rationale_tensor.sum()
            z_ex_nonzero_sum = mask_number.sum().cpu()

            correct += match
            totalp += z_ex_nonzero_sum
            totalr += should_mathch

            precision = match / (z_ex_nonzero_sum + 1e-9)
   
            recall = match/(should_mathch+1e-9)
            F = 2*precision*recall / (precision + recall + 1e-9)
            
            precision_all += precision
            recall_all += recall
            f1_all += F

        test_eval = {}
        test_eval["precision"] = correct / (totalp + 1e-9)
       
        test_eval["recall"] = correct / (totalr + 1e-9)
 
        class_micro_f1, class_macro_precision, class_macro_recall, class_macro_f1 = eval_data_types(true_charge,predictions_charge,num_labels=2)

        test_eval['Acc'] = class_micro_f1
        test_eval['macro_precision'] = class_macro_precision
        test_eval['macro_recall'] = class_macro_recall
        test_eval['macro_f1'] = class_macro_f1
    
        test_s = make_kv_string(test_eval)

        logger.info(" model test {}".format(test_s))

        if epoch > args.epochs-1:
            PATH = args.save_path+args.model_name.lower()+'_a'+str(args.aspect)+'_'+str(epoch)
            torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()

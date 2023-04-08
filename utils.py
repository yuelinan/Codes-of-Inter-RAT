import json
import torch
import numpy as np
from tqdm import tqdm
import random
import os
import pickle
import gzip

def make_kv_string(d):
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)

def get_value(res):
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def gen_result(res, test=False, file_path=None, class_name=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    #print("Micro precision\t%.4f" % micro_precision)
    #print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall) 
    print("Macro f1\t%.4f" % macro_f1)

    return micro_f1, macro_precision, macro_recall,macro_f1

def eval_data_types(target,prediction,num_labels):
    ground_truth_v2 = []
    predictions_v2 = []
    for i in target:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
    
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i] 
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    micro_f1, macro_precision, macro_recall,macro_f1 = gen_result(res)

    return micro_f1, macro_precision, macro_recall,macro_f1


class Data_Process():
    def __init__(self,args):
        self.aspect = args.aspect
        self.data_path = './reviews.260k.train.txt.gz'
        self.dev_data_path = './reviews.260k.heldout.txt.gz'
        self.test_data_path = './annotations.json'
        filepath = './glove.6B.100d.txt.pickle'

        with open(filepath, 'rb') as file:
            embedding, word2id = pickle.load(file)
       
        self.embedding = embedding
        self.word2id = word2id
        
        print('aspect = '+str(self.aspect))
        

    def read_data(self,types):
        random.seed(42)

        if types=='train_all':
            data_all = []
            print('train_data_path:'+self.data_path)
            with gzip.open(self.data_path, 'rt', encoding='utf-8') as f:
                for index,line in enumerate(f):
                    parts = line.split()
                    scores = list(map(float, parts[:5]))
                    if self.aspect > -1:
                        scores = scores[self.aspect]

                    tokens = parts[5:]
                   
                    json_data = {}
                    json_data['fact'] = tokens
                
                    json_data['label'] = scores
                    if scores<0.6 and scores>0.4:continue
                    json_str = json.dumps(json_data)
                    data_all.append(json_str)

            return data_all


        if types=='dev':
            data_all = []
            print('dev_data_path:'+self.dev_data_path)
            with gzip.open(self.dev_data_path, 'rt', encoding='utf-8') as f:
                for index,line in enumerate(f):
                    parts = line.split()
                    scores = list(map(float, parts[:5]))
                    if self.aspect > -1:
                        scores = scores[self.aspect]

                    tokens = parts[5:]
                    
                    json_data = {}
                    json_data['fact'] = tokens
                
                    json_data['label'] = scores
                    if scores<0.6 and scores>0.4:continue
                    json_str = json.dumps(json_data)
                    data_all.append(json_str)
            return data_all

        if types=='test':

            path = './annotations.json'
            data_all = []
            with open(path, mode="r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    tokens = data["x"]
                    scores = data["y"]
                    annotations = [data["0"], data["1"], data["2"],
                                data["3"], data["4"]]
                    if self.aspect > -1:
                        
                        if scores[self.aspect]>=0.6:
                            score = 1
                        elif scores[self.aspect]<=0.4:
                            score = 0
                        else:
                            continue
                        annotations = annotations[self.aspect]
                    json_data = {}
                    
                    json_data['fact'] = tokens

                    json_data['label'] = score
                    
                    json_data['rationale'] = annotations
                    json_str = json.dumps(json_data)
                    data_all.append(json_str)
           
            print(len(data_all))
            return data_all

    def process_data(self,data,types):
       
        if  types=='test':
            fact_all = []
            label = []
            
            for index,line in enumerate(data):
           
                line = json.loads(line)
                fact = line['fact']
                
                rationale = line['rationale']

                label.append(line['label'])
                
                fact_all.append(fact)

            label = torch.tensor(label,dtype=torch.long)
            
            documents,sent_len = self.seq2tensor(fact_all,max_len=300)

            return label,documents,sent_len,rationale
        else:
            fact_all = []
            label = []
            
            for index,line in enumerate(data):
                line = json.loads(line)
                fact = line['fact']

                if line['label']>=0.6:
                    charge = 1
                else:
                    charge = 0
                label.append(charge)
                fact_all.append(fact)
            label = torch.tensor(label,dtype=torch.long)
            
            documents,sent_len = self.seq2tensor(fact_all,max_len=300)

            return label,documents,sent_len


    

    def seq2tensor(self, sents, max_len=350):
        sent_len_max = max([len(s) for s in sents])
        sent_len_max = min(sent_len_max, max_len)

        sent_tensor = torch.LongTensor(len(sents), sent_len_max).zero_()

        sent_len = torch.LongTensor(len(sents)).zero_()
        for s_id, sent in enumerate(sents):
            sent_len[s_id] = len(sent)
            for w_id, word in enumerate(sent):
                if w_id >= sent_len_max: break
                #print(word)
                if word.lower() in self.word2id:
                    sent_tensor[s_id][w_id] = self.word2id[word.lower()] 
                else:
                    sent_tensor[s_id][w_id] = self.word2id['unk_word'] 
        return sent_tensor,sent_len



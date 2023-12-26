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



class Movie_Process():
    def __init__(self,args):
        self.aspect = args.aspect
        self.train_data_path = './train.json'
        # an example of train.json, which we process the ori movie data:
        # {"rationale": [[373, 377], [393, 399], [421, 424], [483, 493], [515, 526], [567, 568], [585, 599]], "text": "veteran actor cl int east wood has never looked as grizz led as he does in true crime , his latest direct orial effort . when st eve e verett ( his newest character ) gets angry at someone , he gl ares them down with those famous dirty h arry eyes , fur rows his brow and f row ns like a grizz ly bear who  's just lost his c ubs . east wood has played some particularly despicable characters in his time , but e verett could just take the cake . he gets my vote , at least , partly because ` ev ' is a drunken affair - a - week woman izer who has many relationship problems , very few of which are with his wife ( d iane ven ora ) . when his colleague at the oak land t ribune is in an ugly car wreck and dies , e verett must take over for her at a vital interview session . the interview is with frank b ea cham (  isa iah w ashington ) , a death - row inmate set to die at midnight for the murder of a pregnant convenience store clerk . east wood fur rows his brow . as e verett gradually finds information , he realizes that b ea cham could very well be innocent . he interviews a key witness ( m ichael j eter ) , who claims that he burst in the door at po cum  's foods because his car had overhe ated , only to see b ea cham standing over the dead woman  's body , blood on his susp enders , gun in hand . but e verett protests : how could he have seen the gun , which was lowered by his side , with the potato chip rack in front of him ? j eter does n 't know what he  's talking about . east wood fur rows his brow . cr ink led expressions and all , cl int is the centre of energy of true crime . the film is by no means a standard action / suspense yarn , but a thoughtful human story in which the characters come before the shoot - outs .  isa iah w ashington has a break - out performance as frank b ea cham , and scenes with him and his weary wife ( l isa gay ham ilton ) are truly heartfelt moments . but the best scenes are ones that feature east wood du king it out with those in authority over him . de nis le ary , as e verett  's editor and boss , has more than a few memorable moments of restrained anger ( you see , ev is sleeping with his wife ) . but hands down , the most enjoyable segments of the film are when j ames woods is on camera . playing the big boss  alan  mann , woods and east wood create amusing chemistry and laugh - out - loud punch lines . when true crime op ts for a high - speed chase to the g over ner  's house at the finale , the quality of film - making takes an abrupt n osed ive . east wood was so successful with colorful character portraits that he did n 't need to switch lanes . true crime is a tension - building , intriguing drama showcase for the talented director and star . this is a road block he could have easily dismissed ( i fur row my brow ) . . . . . . . . . . . . . . . . . . . . . . . . . . . .", "label": 1, "annotation_id": "posR_799.txt"}
        self.dev_data_path = './dev.json'
        self.test_data_path = './test.json'
        filepath = './data/glove.6B.100d.txt.pickle'

        with open(filepath, 'rb') as file:
            embedding, word2id = pickle.load(file)
       
        self.embedding = embedding
        self.word2id = word2id

    def read_data(self,types):
        random.seed(42)

        if types=='train':
            data_all = []
            print('train_data_path:'+self.train_data_path)
            with open(self.train_data_path, mode="r", encoding="utf-8") as f:
                for index,line in enumerate(f):
                    # if index>10:break
                    line = json.loads(line)
                    
                    json_data = {}
                    json_data['fact'] = line['text'].split()
                    json_data['label'] = line['label']
                    json_str = json.dumps(json_data)
                    data_all.append(json_str)

            return data_all

            
        if types=='dev':
            data_all = []
            print('dev_data_path:'+self.dev_data_path)
            with open(self.dev_data_path, mode="r", encoding="utf-8") as f:
                for index,line in enumerate(f):
                    # if index>10:break
                    line = json.loads(line)
 
                    json_data = {}
                    json_data['fact'] = line['text'].split()
                    json_data['label'] = line['label']
                    json_str = json.dumps(json_data)
                    data_all.append(json_str)

            return data_all

        if types=='test':
            
            data_all = []
            with open(self.test_data_path, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = json.loads(line)
                    annotations = line['rationale']

                    json_data = {}
                    
                    json_data['fact'] = line['text'].split()

                    json_data['label'] = line['label']
                    
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
            
            documents,sent_len = self.seq2tensor(fact_all,max_len=1024)

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
            
            documents,sent_len = self.seq2tensor(fact_all,max_len=1024)

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


import torch
import copy
import pickle
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import json
from collections import defaultdict

use_cuda = torch.cuda.is_available()


def custom_collate_fn(batch):
    # input is a list of dialogturn objects
    bt_siz = len(batch)
    # sequence length only affects the memory requirement, otherwise longer is better
    pad_idx, max_seq_len = 39062, 60
    maxturnsnumber = 0
    minturn = 100
    turnsnumbers = []
    e_batch = []
    for i, (d, e, cl_u) in enumerate(batch):
        turnsnumber = len(cl_u)
        if turnsnumber >= maxturnsnumber:
            maxturnsnumber = turnsnumber
        if turnsnumber <= minturn:
            minturn = turnsnumber
    u_batch = []
    u_lens = []
    l_u = []
    max_clu = 0
    for j in range (0, maxturnsnumber):
        u_lensj = np.zeros(bt_siz, dtype = int)
        u_lens.append(u_lensj)
        l_u.append(0)
        u_batchj = []
        e_batchj = []
        for i in range(0, bt_siz):
           u_batchj.append(torch.LongTensor([39062]))
           #if len(batch[1][i]) >= j+1:
           #    e_batchj.append(batch[1][i][j])
           #else:
           #    tmp = []
           #    e_batchj.append(tmp)
        u_batch.append(u_batchj)
        #print (len(u_batch))
        #print (len(u_batch[0]))
        for i, (d, e, cl_u) in enumerate(batch):
            if len(e) >= j+1:
        #        print (e[j])
                e_batchj.append(e[j])
            else:
                tmp = []
                e_batchj.append(tmp)
        e_batch.append(e_batchj)
        #print (len(e_batch))
        #print (len(e_batch[0]))
    for i, (d, e, cl_u)in enumerate(batch):
        turnsnumbers.append(len(cl_u))
        #e_batch.append(e)
        for j,(cl_uj) in enumerate(cl_u):
           cl_u[j] = min(cl_uj, max_seq_len)
           if cl_u[j] > l_u[j]:
               l_u[j] = cl_u[j]
           if cl_u[j] > max_clu:    
               max_clu = cl_u[j]
               
           u_batch[j][i] = torch.LongTensor(d.u[j])
           u_lens[j][i] = cl_u[j]
    t = u_batch.copy()
    for j in range(0, maxturnsnumber):
        u_batch[j] = Variable(torch.ones( bt_siz, max_clu).long() * pad_idx)
    end_tok = torch.LongTensor([2])
    for i in range(bt_siz):
        for j in range (0, maxturnsnumber):
            seq, cur_l = t[j][i], t[j][i].size(0)
       
            if cur_l <= max_clu:
                u_batch[j][i, :cur_l].data.copy_(seq[:cur_l])
            else:
                u_batch[j][i, :].data.copy_(torch.cat((seq[:l_u[j]-1],end_tok),0))      
    sort4utterlength = []
    return u_batch, u_lens, turnsnumbers, max_clu, minturn, e_batch

def get_features_and_adjancency_and_type():
    fats = []
    with open ("TransE2.json") as f:
        load_dict = json.load(f)
        all_entities = load_dict['ent_embeddings.weight']
        all_entities.append([0] * 100)
        #print(all_entities[8784])
        types = load_dict['rel_embeddings.weight']
        entities = np.array(all_entities)
     #   print (entities.shape)
    all_types = [[0 for i in range(100)] for i in range(entities.shape[0])]
    with open ("build_dictionary_final.txt") as f:
        number = 0
        for line in f:
            entities = line.strip().split("\t")
            type1 = entities[2].strip()
            typeall = {"type":0,"country":1,"actor":2,"director":3,"movie":4}
            if entities[1] != number:
                number = entities[1]
                if int(number) < 1 or int(number) >= 8785:
                    print("wrong2")
                
                all_types[int(number)-1] = types[typeall[type1]]
    types = np.array(all_types)
    #print (types.shape)
    adj_lists = defaultdict(set)
    with open ("train2id_final.txt") as f:
        for line in f:
            entities = line.strip().split(" ")
            if len(entities) == 3:
                adj_lists[int(entities[0])].add(int(entities[1]))
                adj_lists[int(entities[1])].add(int(entities[0]))
    #types = np.array(all_types)
    #print (types.shape)
    fats.append(all_entities)
    fats.append(all_types)
    fats.append(adj_lists)
    return fats

class DialogTurn:
    def __init__(self, item):
        self.u = []
        cur_list  = []
        max_word = 0
        for d in item:
            cur_list.append(d)
            if d == 1:
                self.u.append(copy.copy(cur_list))
                cur_list = []
            if d > max_word:
                max_word = d
    def __len__(self):
        length = 0
        for utter in self.u:
            length += len(utter)
        return length

    def __repr__(self):
        strin = ""
        for utter in self.u:
            strin += str(utter)
        return strin


class MovieTriples(Dataset):
    def __init__(self, data_type, length=None):

        if data_type == 'train':
            _file = 'train_entity.dialogues.pkl'
            _file2 = 'train_entity.txt'
        elif data_type == 'valid':
            _file = 'valid_entity.dialogues.pkl'
            _file2 = 'valid_entity.txt'
        elif data_type == 'test':
            _file = 'test_entity.dialogues.pkl'
            _file2 = 'test_entity.txt'
        self.utterance_data = []
        self.entity_data = []

        with open(_file, 'rb') as fp:
            data = pickle.load(fp)
            for d in data:
                self.utterance_data.append(DialogTurn(d))
        with open(_file2, 'r') as f:
            for line in f:
                turnentities = []
                turns = line.split("<s>")
                turns = turns[:-1]
                for turn in turns:
                    #print (turn)
                    entityforturn = []
                    entities = turn.strip().split(" ")
                    for entity in entities:
                        if len(entity):
                            number = int(entity)-10
                            if number < 0 or number >= 8784:
                                print("worng")
                            entityforturn.append(number)
                    turnentities.append(entityforturn)
                #print (turnentities)
                self.entity_data.append(turnentities)    
        # it helps in optimization that the batch be diverse, definitely helps!
        # self.utterance_data.sort(key=cmp_to_key(cmp_dialog))
        if length:
            self.utterance_data = self.utterance_data[2000:2000 + length]
            self.entity_data = self.entity_data[2000:2000 + length]

    def __len__(self):
        return len(self.utterance_data)

    def __getitem__(self, idx):
        dialog = self.utterance_data[idx]
        entity = self.entity_data[idx]
        
        length = []
        for utter in dialog.u:
            length.append(len(utter))
        return dialog, entity, length


def tensor_to_sent(x, inv_dict, gold = False, greedy=False):
    sents = []
    inv_dict[41378] = '<pad>'
    for li in x:
        if not greedy:
            scr = li[1]
            seq = li[0]
        else:
            scr = 0
            seq = li
        sent = []
        if not gold:
            seq = seq[1:]
        for i in seq:
            i = i.item()
            if i in inv_dict:
                sent.append(inv_dict[i])
            else:
                print (i)
            #if i <= 30:
                #print(i)
                #print(inv_dict[i])
            if i == 1:
                break
        sents.append((" ".join(sent), scr))
    return sents

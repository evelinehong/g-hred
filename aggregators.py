import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, adj_lists, cuda=False, gcn=False): 
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()
        self.adj_lists = adj_lists
        self.features = features
        self.cuda = cuda
        self.gcn = gcn
        
    def forward(self, to_neighs, nodes=None, num_sample=15):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        #print (num_sample)
        #to_neighs = set(to_neighs)

        #print (to_neighs)
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, 
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs
        #print (samp_neighs)
        if self.gcn:
            if not nodes is None:
                samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        #if len(samp_neighs):
        #print (samp_neighs)
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]   
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh).cuda()
        if not nodes is None:
            if self.cuda:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
            else:
                embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        else:
            #adj_adj_lists = []
            #for to_neigh in to_neighs:
            #    adjs2 = [self.adj_lists[int(node)] for node in to_neigh]
            #    if not len(adjs2):
            #        adjs2.append(8784)
            #    print(adjs2)
            #    adj_adj_lists.append(adjs2) 
                
            adj_adj_lists = [self.adj_lists[int(node)] for node in unique_nodes_list]
            #print (adj_adj_lists)
            adj_adj_lists2 = []
            for adj_adj_list in adj_adj_lists:
                if not len (adj_adj_list) or adj_adj_list==set():
                    adj_adj_list = {8784}
                adj_adj_lists2.append(adj_adj_list)
       
            #print (adj_adj_lists2)
            if self.cuda:
                embed_matrix = self.features(adj_adj_lists2, torch.LongTensor(unique_nodes_list).cuda())
            else:
                embed_matrix = self.features(adj_adj_lists2, torch.LongTensor(unique_nodes_list))
            
        to_feats = mask.mm(embed_matrix)
        #print (to_feats)
        return to_feats

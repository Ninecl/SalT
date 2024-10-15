import os
import time
import random
import logging

import numpy as np

from scipy.sparse import csc_matrix, coo_matrix
from torch.utils.data import Dataset

from utils import readTriplets2Id


class GraphDataset(Dataset):
    
    def __init__(self, params) -> None:
        super().__init__()
        
        self.ori_data_path = os.path.join("./data", params.dataset, "ori")
        self.emg_data_path = os.path.join("./data", params.dataset, "emg")
        self.entity2id = dict()
        self.relation2id = dict()
        self.ori_train_path = os.path.join(self.ori_data_path, "train.txt")
        self.ori_valid_path = os.path.join(self.ori_data_path, "valid.txt")
        self.ori_test_path = os.path.join(self.ori_data_path, "test.txt")
        
        self.ori_train, self.entity2id, self.relation2id = readTriplets2Id(self.ori_train_path, 'hrt', self.entity2id, self.relation2id)
        self.ori_valid = readTriplets2Id(self.ori_valid_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.ori_test = readTriplets2Id(self.ori_test_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        self.ori_all = self.ori_train + self.ori_valid + self.ori_test
        
        self.num_ent = len(self.entity2id)
        self.num_rel = len(self.relation2id)
        # fliter dic
        self.filter_dic = self.get_filter(self.ori_all)
        
        
    def split(self, support_percent):
        ent_set = set()
        rel_set = set()
        support_triplet_idxs = []
        train_triplet_idxs = np.arange(len(self.ori_train))
        np.random.shuffle(train_triplet_idxs)
        
        for idx in train_triplet_idxs:
            h, r, t = self.ori_train[idx]
            if (h not in ent_set) or (t not in ent_set) or (r not in rel_set):
                ent_set.add(h)
                ent_set.add(t)
                rel_set.add(r)
                support_triplet_idxs.append(idx)
                
        num_support = int(len(self.ori_train) * support_percent)
        remained_triplet_idxs = list(set(train_triplet_idxs) - set(support_triplet_idxs))
        np.random.shuffle(remained_triplet_idxs)
        split_idx = num_support - len(support_triplet_idxs)
        support_triplet_idxs += remained_triplet_idxs[: split_idx]
        query_triplet_idxs = remained_triplet_idxs[split_idx: ]
            
        support_triplets = np.array(self.ori_train)[support_triplet_idxs]
        query_triplets = np.array(self.ori_train)[query_triplet_idxs]
        
        # print(support_triplets)
        # print(query_triplets)
        
        adj_matrix = csc_matrix((np.ones(len(support_triplets)), (support_triplets[:, 0], support_triplets[:, 2])), 
                                shape=(self.num_ent, self.num_ent)).tocoo()
        row_idxs = adj_matrix.row
        col_idxs = adj_matrix.col
        adj_matrix = coo_matrix((np.ones(len(row_idxs)), (row_idxs, col_idxs)), 
                                shape=(self.num_ent, self.num_ent)).todense()
        
        # relational adjacent matrices
        relational_adj_matrices = np.zeros((self.num_rel, self.num_ent, self.num_ent))
        # entity relational feature
        ent_relational_fearues = np.zeros((self.num_ent, self.num_rel * 2))
        for h, r, t in support_triplets:
            relational_adj_matrices[r][h][t] += 1
            ent_relational_fearues[h][r] += 1
            ent_relational_fearues[t][r + self.num_rel] += 1

        return ent_relational_fearues, adj_matrix, relational_adj_matrices, query_triplets

    
    def update(self):
        emg_sup_path = os.path.join(self.emg_data_path, 'support.txt')
        emg_que_path = os.path.join(self.emg_data_path, 'query.txt')
        
        self.emg_sup, self.entity2id, self.relation2id = readTriplets2Id(emg_sup_path, 'hrt', self.entity2id, self.relation2id)
        self.emg_que = readTriplets2Id(emg_que_path, 'hrt', self.entity2id, self.relation2id, allow_emerging=False)
        
        self.emg_all = self.emg_sup + self.ori_train
        self.num_rel = len(self.relation2id)
        self.num_ent = len(self.entity2id)
        
        all_triplets = self.ori_train + self.ori_valid + self.ori_test + self.emg_sup + self.emg_que
        self.filter_dic = self.get_filter(all_triplets)
        
        return self.num_ent
    
    
    def get_filter(self, triplets):
        fliter_dic = {}
        
        for triplet in triplets:
            h, r, t = triplet
            if (h, r, '_') not in fliter_dic:
                fliter_dic[(h, r, '_')] = [t, ]
            else:
                fliter_dic[(h, r, '_')].append(t)
            if (h, '_', t) not in fliter_dic:
                fliter_dic[(h, '_', t)] = [r, ]
            else:
                fliter_dic[(h, '_', t)].append(r)
            if ('_', r, t) not in fliter_dic:
                fliter_dic[('_', r, t)] = [h, ]
            else:
                fliter_dic[('_', r, t)].append(h)
                
        return fliter_dic

    
    def get_ent_features_and_adj(self, support_triplets):
        adj_matrix = csc_matrix((np.ones(len(support_triplets)), (support_triplets[:, 0], support_triplets[:, 2])), 
                                shape=(self.num_ent, self.num_ent)).tocoo()
        row_idxs = adj_matrix.row
        col_idxs = adj_matrix.col
        adj_matrix = coo_matrix((np.ones(len(row_idxs)), (row_idxs, col_idxs)), 
                                shape=(self.num_ent, self.num_ent)).todense()
        
        # relational adjacent matrices
        relational_adj_matrices = np.zeros((self.num_rel, self.num_ent, self.num_ent))
        # entity relational feature
        ent_relational_fearues = np.zeros((self.num_ent, self.num_rel * 2))
        for h, r, t in support_triplets:
            relational_adj_matrices[r][h][t] += 1
            ent_relational_fearues[h][r] += 1
            ent_relational_fearues[t][r + self.num_rel] += 1

        return ent_relational_fearues, adj_matrix, relational_adj_matrices
    

    def __len__(self):
        return len(self.ori_train)
    
    
    def __getitem__(self, index):
        return self.ori_train[index]
import torch
import logging

import numpy as np

from tqdm import tqdm
from utils import move_test_data_to_device, get_batch_ent_features_and_adj


def evaluate(my_model, data, device, mode='valid', limited_candidate=None):
    
    with torch.no_grad():
        # 1. check mode
        if mode == 'valid':
            # get support triplets and forward model
            support_triplets = np.array(data.ori_train)
            ent_fearues, adj, rel_adjs = get_batch_ent_features_and_adj(support_triplets, data.num_ent, data.num_rel)
            ent_fearues, ent_idxs, rel_adjs = move_test_data_to_device(ent_fearues, rel_adjs, device)
            ent_embs = my_model(ent_fearues, ent_idxs, rel_adjs, 0)
            # get query triplets
            query_triplets = torch.tensor(data.ori_valid).cuda()
        elif mode == 'trans':
            support_triplets = np.array(data.ori_train)
            ent_fearues, adj, rel_adjs = get_batch_ent_features_and_adj(support_triplets, data.num_ent, data.num_rel)
            ent_fearues, ent_idxs, rel_adjs = move_test_data_to_device(ent_fearues, rel_adjs, device)
            ent_embs = my_model(ent_fearues, ent_idxs, rel_adjs, 0)
            # get query triplets
            query_triplets = torch.tensor(data.ori_test).cuda()
        elif mode == 'ind' or mode == 'IT':
            num_ent = data.update()
            my_model.update(num_ent, device)
            support_triplets = np.array(data.emg_all)
            ent_fearues, adj, rel_adjs = get_batch_ent_features_and_adj(support_triplets, data.num_ent, data.num_rel)
            ent_fearues, ent_idxs, rel_adjs = move_test_data_to_device(ent_fearues, rel_adjs, device)
            ent_embs = my_model(ent_fearues, ent_idxs, rel_adjs, 0)
            # get query triplets
            if mode == 'ind':
                query_triplets = torch.tensor(data.emg_que).cuda()
            elif mode == 'IT':
                query_triplets = torch.tensor(data.ori_test).cuda()
        # 2. calculate score and ranks
        head_ranks = []
        tail_ranks = []
        ranks = []
        for triplet in tqdm(query_triplets):
            # 3. get one query triplet
            h, r, t = triplet
            # 4. head corrupt
            head_corrupt = triplet.unsqueeze(dim=0).repeat(data.num_ent, 1)
            head_corrupt[:, 0] = torch.arange(end=data.num_ent)
            # 5. get head rank
            head_scores = my_model.score(head_corrupt, ent_embs)
            head_filters = data.filter_dic[('_', int(r), int(t))]
            head_rank = get_rank(triplet, head_scores, head_filters, limited_candidate, target=0)
            # 6. tail corrupt
            tail_corrupt = triplet.unsqueeze(dim=0).repeat(data.num_ent, 1)
            tail_corrupt[:, 2] = torch.arange(end=data.num_ent)

            tail_scores = my_model.score(tail_corrupt, ent_embs)
            tail_filters = data.filter_dic[(int(h), int(r), '_')]
            tail_rank = get_rank(triplet, tail_scores, tail_filters, limited_candidate, target=2)

            ranks.append(head_rank)
            head_ranks.append(head_rank)
            ranks.append(tail_rank)
            tail_ranks.append(tail_rank)
            # print(head_rank, tail_rank)


        logging.info(f"=========={mode} LP==========")
        a_mr, a_mrr, a_hit1, a_hit3, a_hit10 = get_metrics(ranks)
        h_mr, h_mrr, h_hit1, h_hit3, h_hit10 = get_metrics(head_ranks)
        t_mr, t_mrr, t_hit1, t_hit3, t_hit10 = get_metrics(tail_ranks)
        logging.info(f"A | MR: {a_mr:.1f} | MRR: {a_mrr:.3f} | Hits@1: {a_hit1:.3f} | Hits@3: {a_hit3:.3f} | Hits@10: {a_hit10:.3f}")
        logging.info(f"H | MR: {h_mr:.1f} | MRR: {h_mrr:.3f} | Hits@1: {h_hit1:.3f} | Hits@3: {h_hit3:.3f} | Hits@10: {h_hit10:.3f}")
        logging.info(f"T | MR: {t_mr:.1f} | MRR: {t_mrr:.3f} | Hits@1: {t_hit1:.3f} | Hits@3: {t_hit3:.3f} | Hits@10: {t_hit10:.3f}")
        
    return a_mr, a_mrr, a_hit10, a_hit3, a_hit1

    
def get_rank(triplet, scores, filters, limited_candidate, target=0):
    thres = scores[triplet[target]].item()
    scores[filters] = thres - 1
    if limited_candidate is None:
        rank = (scores > thres).sum() + 1
    else:
        scores = np.random.choice(scores.cpu(), limited_candidate)
        rank = (scores > thres).sum() + 1
    return rank.item()


def get_metrics(rank):
	rank = np.array(rank)
	mr = np.mean(rank)
	mrr = np.mean(1 / rank)
	hit10 = np.sum(rank < 11) / len(rank)
	hit3 = np.sum(rank < 4) / len(rank)
	hit1 = np.sum(rank < 2) / len(rank)
	return mr, mrr, hit1, hit3, hit10
import os


DATASET = 'CoDEX'


def readTriplets(path, mode, with_head=False):

    triplets = []
    with open(path, 'r') as f:
        data = f.readlines() if not with_head else f.readlines()[1: ]
        lines = [line.strip().split() for line in data]
        for line in lines:
            if mode == 'hrt':
                h, r, t = line
            elif mode == 'htr':
                h, t, r = line
            else:
                raise "ERROR: illegal triplet form"
            triplets.append((h, r, t))
    return triplets


def triplets2id(triplets):
    ent2id, rel2id = dict(), dict()
    ent_cnt, rel_cnt = 0, 0
    for h, r, t in triplets:
        if h not in ent2id:
            ent2id[h] = ent_cnt
            ent_cnt += 1
        if t not in ent2id:
            ent2id[t] = ent_cnt
            ent_cnt += 1
        if r not in rel2id:
            rel2id[r] = rel_cnt
            rel_cnt += 1
    return ent2id, rel2id


ori_train_path = f'./{DATASET}/ori/train.txt'
ori_valid_path = f'./{DATASET}/ori/valid.txt'
ori_test_path = f'./{DATASET}/ori/test.txt'

ori_train_triples = readTriplets(ori_train_path, 'hrt')
ori_valid_triples = readTriplets(ori_valid_path, 'hrt')
ori_test_triples = readTriplets(ori_test_path, 'hrt')
ori_all_triplets = ori_train_triples + ori_valid_triples + ori_test_triples

ori_ent2id, ori_rel2id = triplets2id(ori_all_triplets)
print("There {} entities, {} relations, {} train_triplets, {} valid triplets, and {} test triplets in ori KG.".format(len(ori_ent2id), len(ori_rel2id), len(ori_train_triples), len(ori_valid_triples), len(ori_test_triples)))


emg_support_path = f'./{DATASET}/emg/support.txt'
emg_query_path = f'./{DATASET}/emg/query.txt'

emg_support_triplets = readTriplets(emg_support_path, 'hrt')
emg_query_triplets = readTriplets(emg_query_path, 'hrt')
all_support_triplets = ori_train_triples + emg_support_triplets

emg_ent2id, emg_rel2id = triplets2id(all_support_triplets)
print("There {} entities, {} relations, {} train_triplets, and {} test triplets in ori KG.".format(len(emg_ent2id), len(emg_rel2id), len(all_support_triplets), len(emg_query_triplets)))





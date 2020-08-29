import json
from os.path import join
import numpy as np
import torch
from torch.utils.data import Dataset
from keras.preprocessing.text import Tokenizer

from utils import data_utils
from utils import settings


def gen_all_domain_tokenizer():
    tokenizer = Tokenizer(num_words=settings.MAX_WORD_TOKEN_NUM)
    corpus = []

    print("building aff corpus...")

    # aff
    file_dir = settings.AFF_DATA_DIR
    pos_pairs = data_utils.load_json(file_dir, "label_data_aff_zhoushao.json")[:600]
    pos_pairs = [({"name": p["affiliation"]}, {"DisplayName": p["label"]}) for p in pos_pairs if p["label"] != "[NIF]"]
    neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi_clean.json')[:600]
    neg_pairs = [(p['aminer_affi'], p['mag_affi']) for p in neg_pairs]
    pairs_add = data_utils.load_json(file_dir, "mag_aminer_hard_correct_zfj_copy.json")
    print("add pairs", len(pairs_add))
    pos_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "1"]
    neg_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "0"]
    pairs = pos_pairs + neg_pairs

    for item in pairs:
        corpus.append(item[0]["name"].lower())
        corpus.append(item[1]["DisplayName"].lower())

    print("building author corpus...")

    # author
    file_dir = settings.AUTHOR_DATA_DIR
    pos_pairs = data_utils.load_json(file_dir, 'pos_person_pairs.json')
    neg_pairs = data_utils.load_json(file_dir, 'neg_person_pairs.json')
    pairs = pos_pairs + neg_pairs

    person_dict = data_utils.load_json(file_dir, "ego_person_dict.json")

    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            print("author process pair", i)
        # cpaper, npaper = pair
        aid, mid = pair['aid'], pair['mid']
        aperson = person_dict.get(aid, {})
        mperson = person_dict.get(mid, {})
        corpus.append(aperson.get("pubs", []))
        corpus.append(mperson.get("pubs", []))

        corpus.append([item["id"] for item in aperson.get("venue", [])])
        corpus.append([item["id"] for item in mperson.get("venue", [])])

    print("building paper corpus...")

    # paper
    file_dir = settings.PAPER_DATA_DIR
    pos_pairs = data_utils.load_json(file_dir, 'pos-pairs-train.json')
    pos_pairs = [(p['c'], p['n']) for p in pos_pairs]
    neg_pairs = data_utils.load_json(file_dir, 'neg-pairs-train.json')
    neg_pairs = [(p['c'], p['n']) for p in neg_pairs]
    pairs = pos_pairs + neg_pairs

    for i, pair in enumerate(pairs):
        if i % 100 == 0:
            print('pairs to matrices', i)
        cpaper, npaper = pair
        corpus.append(cpaper["title"])
        corpus.append(npaper["title"])
        corpus.append(" ".join(cpaper["authors"]))
        corpus.append(" ".join(npaper["authors"]))

    print("building venue corpus...")

    train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train.txt'), 'r'))
    for item in train_data:
        corpus.append(item[1].lower())
        corpus.append(item[2].lower())

    tokenizer.fit_on_texts(corpus)
    vocab_size = len(tokenizer.word_counts)
    print("vocab size", vocab_size)

    data_utils.dump_large_obj(tokenizer, settings.OUT_DIR, "tokenizer_all_domain.pkl")
    emb = torch.randn(size=(settings.MAX_WORD_TOKEN_NUM + 1, 128))
    print("emb", emb)
    torch.save(emb, join(settings.OUT_DIR, "rnn_init_word_emb.emb"))


class ProcessedCNNInputDataset(Dataset):

    def __init__(self, entity_type, role, sample_num=None):

        data_dir = settings.DOM_ADAPT_DIR
        fname = "{}_{}.pkl".format(entity_type, role)
        data_dict = data_utils.load_large_obj(data_dir, fname)
        self.x1 = np.array(data_dict["x1"], dtype="float32")
        self.x2 = np.array(data_dict["x2"], dtype="float32")
        self.y = np.array(data_dict["y"], dtype=int)

        self.N = len(self.y)

        if sample_num is None:
            sample_num = self.N

        self.x1 = torch.from_numpy(self.x1)[:sample_num]
        self.x2 = torch.from_numpy(self.x2)[:sample_num]
        self.y = torch.from_numpy(self.y)[:sample_num]

        self.N = sample_num

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]


class ProcessedRNNInputDataset(Dataset):

    def __init__(self, entity_type, role, sample_num=None):

        data_dir = settings.DOM_ADAPT_DIR
        fname = "{}_rnn_{}.pkl".format(entity_type, role)
        data_dict = data_utils.load_large_obj(data_dir, fname)
        self.x1_seq1 = np.array(data_dict["x1_seq1"], dtype=int)
        self.x1_seq2 = np.array(data_dict["x1_seq2"], dtype=int)
        self.x2_seq1 = np.array(data_dict["x2_seq1"], dtype=int)
        self.x2_seq2 = np.array(data_dict["x2_seq2"], dtype=int)
        self.y = np.array(data_dict["y"], dtype=int)

        self.N = len(self.y)

        if sample_num is not None:
            n_sample_half = int(sample_num/2)
            pos_flag = False
            neg_flag = False
            x1_1 = []
            x1_2 = []
            x2_1 = []
            x2_2 = []
            y = []
            n_pos = 0
            n_neg = 0
            for i in range(self.N):
                if pos_flag and neg_flag:
                    break
                cur_y = self.y[i]
                if cur_y == 1 and n_pos < n_sample_half:
                    x1_1.append(self.x1_seq1[i])
                    x1_2.append(self.x1_seq2[i])
                    x2_1.append(self.x2_seq1[i])
                    x2_2.append(self.x2_seq2[i])
                    y.append(cur_y)
                    n_pos += 1
                    if n_pos == n_sample_half:
                        pos_flag = True
                elif cur_y == 0 and n_neg < n_sample_half:
                    x1_1.append(self.x1_seq1[i])
                    x1_2.append(self.x1_seq2[i])
                    x2_1.append(self.x2_seq1[i])
                    x2_2.append(self.x2_seq2[i])
                    y.append(cur_y)
                    n_neg += 1
                    if n_neg == n_sample_half:
                        neg_flag = True
            self.x1_seq1 = np.array(x1_1)
            self.x1_seq2 = np.array(x1_2)
            self.x2_seq1 = np.array(x2_1)
            self.x2_seq2 = np.array(x2_2)
            self.y = np.array(y)

        self.x1_seq1 = torch.from_numpy(self.x1_seq1)
        self.x1_seq2 = torch.from_numpy(self.x1_seq2)
        self.x2_seq1 = torch.from_numpy(self.x2_seq1)
        self.x2_seq2 = torch.from_numpy(self.x2_seq2)
        self.y = torch.from_numpy(self.y)

        self.N = len(self.y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x1_seq1[idx], self.x2_seq1[idx], self.x1_seq2[idx], self.x2_seq2[idx], self.y[idx]


if __name__ == "__main__":
    gen_all_domain_tokenizer()

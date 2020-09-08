from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import argparse
import os
from os.path import join
from collections import defaultdict as dd
import numpy as np
import sklearn
import torch
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences

from utils import feature_utils
from utils import data_utils
from utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


class AuthorCNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle, args, use_emb=True):
        self.file_dir = file_dir
        self.matrix_title_size = matrix_size1
        self.matrix_author_size = matrix_size2

        # load training pairs
        pos_pairs = data_utils.load_json(file_dir, 'pos_person_pairs.json')
        neg_pairs = data_utils.load_json(file_dir, 'neg_person_pairs.json')
        pairs = pos_pairs + neg_pairs
        labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        self.person_dict = data_utils.load_json(file_dir, "ego_person_dict.json")

        self.use_emb = use_emb
        if self.use_emb:
            self.pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))
        self.tokenizer = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        X_long = []
        X_short = []
        nn_pos = 0
        nn_neg = 0
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d %d %d', i, nn_pos, nn_neg)
            aid, mid = pair['aid'], pair['mid']
            aperson = self.person_dict.get(aid, {})
            mperson = self.person_dict.get(mid, {})
            # matrix1, nn1 = self.org_to_matrix(aperson.get('org', ''), mperson.get('org', ''), matrix_size1)
            matrix1, nn1 = self.paper_to_matrix(aperson.get('pubs', []), mperson.get('pubs', []), matrix_size1)
            matrix1 = feature_utils.scale_matrix(matrix1)
            X_long.append(matrix1)
            matrix2, nn2 = self.venue_to_matrix(aperson.get('venue', ''), mperson.get('venue', ''), matrix_size2)
            # print("matrix2", matrix2)
            matrix2 = feature_utils.scale_matrix(matrix2)
            X_short.append(matrix2)

        self.X_long = X_long
        self.X_short = X_short
        self.Y = labels

        print("shuffle", shuffle)
        if shuffle:
            self.X_long, self.X_short, self.Y = sklearn.utils.shuffle(
                self.X_long, self.X_short, self.Y,
                random_state=seed
            )

        self.N = len(self.Y)

        # valid_start = int(self.N * args.train_ratio / 100)
        # test_start = int(self.N * (args.train_ratio + args.valid_ratio) / 100)
        valid_start = 800
        test_start = 100 + valid_start
        end_point = 100 + test_start

        train_data = {}
        train_data["x1"] = self.X_long[:valid_start]
        train_data["x2"] = self.X_short[:valid_start]
        train_data["y"] = self.Y[:valid_start]
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1"] = self.X_long[test_start: end_point]
        test_data["x2"] = self.X_short[test_start: end_point]
        test_data["y"] = self.Y[test_start: end_point]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1"] = self.X_long[valid_start:test_start]
        valid_data["x2"] = self.X_short[valid_start:test_start]
        valid_data["y"] = self.Y[valid_start:test_start]
        print("valid labels", len(valid_data["y"]))

        print("train positive samples", sum(train_data["y"]))
        print("test positive samples", sum(test_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "author_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "author_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "author_valid.pkl")

    def paper_to_matrix(self, ap, mp, max_size):
        if self.use_emb:  #TODO
            apubs = self.tokenizer.texts_to_sequences([" ".join(ap)])[0][:max_size]
            mpubs = self.tokenizer.texts_to_sequences([" ".join(mp)])[0][:max_size]
        else:
            apubs = ap[:max_size]
            mpubs = mp[:max_size]

        matrix = -np.ones((max_size, max_size))
        for i, apid in enumerate(apubs):
            for j, mpid in enumerate(mpubs):
                v = -1
                if apid == mpid:
                    v = 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[apid].reshape(1, -1),
                                      self.pretrain_emb[mpid].reshape(1, -1))[0][0]
                matrix[i, j] = v
        return matrix, None

    def venue_to_matrix(self, o1, o2, max_size):
        avenue = [v['id'] for v in o1]
        mvenue = [v['id'] for v in o2]
        if self.use_emb:
            avenue = self.tokenizer.texts_to_sequences([" ".join(avenue)])[0][:max_size]
            mvenue = self.tokenizer.texts_to_sequences([" ".join(mvenue)])[0][:max_size]
        avenue = avenue[: max_size]
        mvenue = mvenue[: max_size]

        matrix = -np.ones((max_size, max_size))
        nn1 = 0
        for i, avid in enumerate(avenue):
            for j, mvid in enumerate(mvenue):
                v = -1
                if avid == mvid:
                    v = 1
                    nn1 += 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[avid].reshape(1, -1),
                                      self.pretrain_emb[mvid].reshape(1, -1))[0][0]
                matrix[i, j] = v
        print(nn1, avenue, mvenue)
        return matrix, None


class AuthorRNNMatchDataset(Dataset):

    def __init__(self, file_dir, max_seq1_len, max_seq2_len, shuffle, seed, args):

        self.max_seq1_len = max_seq1_len
        self.max_seq2_len = max_seq2_len

        # load training pairs
        pos_pairs = data_utils.load_json(file_dir, 'pos_person_pairs.json')
        neg_pairs = data_utils.load_json(file_dir, 'neg_person_pairs.json')
        pairs = pos_pairs + neg_pairs
        self.labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)

        self.person_dict = data_utils.load_json(file_dir, "ego_person_dict.json")
        nn_pos = 0
        nn_neg = 0
        t = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        self.vocab_size = len(t.word_counts)
        print("vocab size", self.vocab_size)

        self.mag = [self.person_dict.get(pair["mid"], {}).get("pubs", []) for pair in pairs]
        self.aminer = [self.person_dict.get(pair["aid"], {}).get("pubs", []) for pair in pairs]
        self.mag = t.texts_to_sequences(self.mag)

        self.aminer = t.texts_to_sequences(self.aminer)
        self.mag = pad_sequences(self.mag, maxlen=self.max_seq1_len)
        self.aminer = pad_sequences(self.aminer, maxlen=self.max_seq1_len)

        self.mag_keywords = []
        self.aminer_keywords = []
        for i, pair in enumerate(pairs):
            if i % 100 == 0:
                logger.info('pairs to matrices %d %d %d', i, nn_pos, nn_neg)
            aid, mid = pair['aid'], pair['mid']
            avenue = [item["id"] for item in self.person_dict.get(aid, {}).get("venue", [])]
            mvenue = [item["id"] for item in self.person_dict.get(mid, {}).get("venue", [])]
            self.mag_keywords.append(mvenue)
            self.aminer_keywords.append(avenue)

        self.mag_keywords = t.texts_to_sequences(self.mag_keywords)
        self.aminer_keywords = t.texts_to_sequences(self.aminer_keywords)

        self.mag_keywords = pad_sequences(self.mag_keywords, maxlen=max_seq2_len)
        self.aminer_keywords = pad_sequences(self.aminer_keywords, maxlen=max_seq2_len)

        if shuffle:
            self.mag, self.aminer, self.mag_keywords, self.aminer_keywords, self.labels = sklearn.utils.shuffle(
                self.mag, self.aminer, self.mag_keywords, self.aminer_keywords, self.labels,
                random_state=seed
            )

        self.N = len(self.labels)

        # valid_start = int(self.N * args.train_ratio / 100)
        # test_start = int(self.N * (args.train_ratio + args.valid_ratio) / 100)
        valid_start = 800
        test_start = 100 + valid_start
        end_point = 100 + test_start

        train_data = {}
        train_data["x1_seq1"] = self.mag[:valid_start]
        train_data["x1_seq2"] = self.mag_keywords[:valid_start]
        train_data["x2_seq1"] = self.aminer[:valid_start]
        train_data["x2_seq2"] = self.aminer_keywords[:valid_start]
        train_data["y"] = self.labels[:valid_start]
        train_data["vocab_size"] = self.vocab_size
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1_seq1"] = self.mag[test_start: end_point]
        test_data["x1_seq2"] = self.mag_keywords[test_start: end_point]
        test_data["x2_seq1"] = self.aminer[test_start: end_point]
        test_data["x2_seq2"] = self.aminer_keywords[test_start: end_point]
        test_data["y"] = self.labels[test_start: end_point]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1_seq1"] = self.mag[valid_start:test_start]
        valid_data["x1_seq2"] = self.mag_keywords[valid_start:test_start]
        valid_data["x2_seq1"] = self.aminer[valid_start:test_start]
        valid_data["x2_seq2"] = self.aminer_keywords[valid_start:test_start]
        valid_data["y"] = self.labels[valid_start:test_start]
        print("valid labels", len(valid_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "author_rnn_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "author_rnn_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "author_rnn_valid.pkl")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default=settings.AUTHOR_DATA_DIR, help="Input file directory")
    parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
    parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
    parser.add_argument('--train-ratio', type=int, default=50, help='Training size.')
    parser.add_argument('--valid-ratio', type=int, default=25, help='Testing size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
    parser.add_argument('--max-sequence-length', type=int, default=17,
                        help="Max sequence length for raw sequences")
    parser.add_argument('--max-key-sequence-length', type=int, default=8,
                        help="Max key sequence length for key sequences")
    args = parser.parse_args()
    dataset = AuthorCNNMatchDataset(file_dir=args.file_dir, matrix_size1=args.matrix_size1, matrix_size2=args.matrix_size2, seed=args.seed, shuffle=True, args=args, use_emb=False)
    dataset = AuthorRNNMatchDataset(args.file_dir, args.max_sequence_length, args.max_key_sequence_length, shuffle=True, seed=args.seed, args=args)

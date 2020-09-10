from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from os.path import join
import sklearn
import json
import codecs
import _pickle
import nltk
import argparse
from torch.utils.data import Dataset
import torch
from sklearn.metrics.pairwise import cosine_similarity

from keras.preprocessing.text import text
from keras.preprocessing.sequence import pad_sequences

from utils import feature_utils
from utils import data_utils
from utils import settings

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


def filter_venue_dataset():
    train_data = json.load(open(join(settings.VENUE_DATA_DIR, "train_copy_zfj.txt")))
    ddd = {}
    train_data_new = []
    for pair in train_data:
        cur = sorted(pair[1:])
        cur_key = "&&&".join(cur)

        if cur_key in ddd:
            print(cur_key)
        else:
            ddd[cur_key] = 1
            train_data_new.append(pair)
    print("size after filtering", len(train_data_new))
    data_utils.dump_json(train_data_new, settings.VENUE_DATA_DIR, "train_filter.txt")


class VenueCNNMatchDataset(Dataset):

    def __init__(self, file_dir, matrix_size1, matrix_size2, seed, shuffle, args, use_emb=True):
        self.file_dir = file_dir

        self.matrix_size_1_long = matrix_size1
        self.matrix_size_2_short = matrix_size2

        self.use_emb = use_emb
        if self.use_emb:
            self.pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))
        self.tokenizer = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        self.train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train_filter.txt'), 'r'))

        n_pos_set = int((args.train_num + 2 * args.test_num)/2)

        neg_pairs = [p for p in self.train_data if p[0] == 0]
        pos_pairs = [p for p in self.train_data if p[0] == 1][-n_pos_set:]
        n_pos = len(pos_pairs)
        neg_pairs = neg_pairs[-n_pos:]
        self.train_data = pos_pairs + neg_pairs

        self.train_data = sklearn.utils.shuffle(self.train_data, random_state=37)

        self.mag = [nltk.word_tokenize(p[1]) for p in self.train_data]
        self.aminer = [nltk.word_tokenize(p[2]) for p in self.train_data]
        self.labels = [p[0] for p in self.train_data]

        self.calc_keyword_seqs()

        n_matrix = len(self.labels)
        self.X_long = np.zeros((n_matrix, self.matrix_size_1_long, self.matrix_size_1_long))
        self.X_short = np.zeros((n_matrix, self.matrix_size_2_short, self.matrix_size_2_short))
        self.Y = np.zeros(n_matrix, dtype=np.long)
        count = 0
        for i, cur_y in enumerate(self.labels):
            if i % 100 == 0:
                print('pairs to matrices', i)
            v1 = self.mag[i]
            v1 = " ".join([str(v) for v in v1])
            v2 = self.aminer[i]
            v2 = " ".join([str(v) for v in v2])
            v1_key = self.mag_venue_keywords[i]
            v1_key = " ".join([str(v) for v in v1_key])
            v2_key = self.aminer_venue_keywords[i]
            v2_key = " ".join([str(v) for v in v2_key])
            matrix1 = self.sentences_long_to_matrix(v1, v2)
            # print("mat1", matrix1)
            self.X_long[count] = feature_utils.scale_matrix(matrix1)
            matrix2 = self.sentences_short_to_matrix(v1_key, v2_key)
            # print("mat2", matrix2)
            self.X_short[count] = feature_utils.scale_matrix(matrix2)
            self.Y[count] = cur_y
            count += 1

        self.N = len(self.Y)

        n_train = args.train_num
        n_test = args.test_num
        # n_train = self.N - 2*n_test

        train_data = {}
        train_data["x1"] = self.X_long[:n_train]
        train_data["x2"] = self.X_short[:n_train]
        train_data["y"] = self.Y[:n_train]
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1"] = self.X_long[n_train:(n_train+n_test)]
        test_data["x2"] = self.X_short[n_train:(n_train+n_test)]
        test_data["y"] = self.Y[n_train:(n_train+n_test)]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1"] = self.X_long[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2"] = self.X_short[n_train+n_test:(n_train+n_test*2)]
        valid_data["y"] = self.Y[n_train+n_test:(n_train+n_test*2)]
        print("valid labels", len(valid_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "venue_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "venue_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "venue_valid.pkl")

    def calc_keyword_seqs(self):
        N = len(self.mag)
        mag_keywords = []
        aminer_keywords = []
        for i in range(N):
            cur_v_mag = self.mag[i]
            cur_v_aminer = self.aminer[i]
            overlap = set(cur_v_mag).intersection(cur_v_aminer)
            new_seq_mag = []
            new_seq_aminer = []
            for w in cur_v_mag:
                if w in overlap:
                    new_seq_mag.append(w)
            for w in cur_v_aminer:
                if w in overlap:
                    new_seq_aminer.append(w)
            mag_keywords.append(new_seq_mag)
            aminer_keywords.append(new_seq_aminer)
        self.mag_venue_keywords = mag_keywords
        self.aminer_venue_keywords = aminer_keywords

    def sentences_long_to_matrix(self, title1, title2):
        if self.use_emb:
            twords1 = self.tokenizer.texts_to_sequences([title1])[0][: self.matrix_size_1_long]
            twords2 = self.tokenizer.texts_to_sequences([title2])[0][: self.matrix_size_1_long]
        else:
            twords1 = feature_utils.get_words(title1)[: self.matrix_size_1_long]
            twords2 = feature_utils.get_words(title2)[: self.matrix_size_1_long]

        matrix = -np.ones((self.matrix_size_1_long, self.matrix_size_1_long))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                v = -1
                if word1 == word2:
                    v = 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[word1].reshape(1, -1),
                                          self.pretrain_emb[word2].reshape(1, -1))[0][0]
                matrix[i][j] = v
        return matrix

    def sentences_short_to_matrix(self, title1, title2):
        if self.use_emb:
            twords1 = self.tokenizer.texts_to_sequences([title1])[0][: self.matrix_size_2_short]
            twords2 = self.tokenizer.texts_to_sequences([title2])[0][: self.matrix_size_2_short]
        else:
            twords1 = feature_utils.get_words(title1)[: self.matrix_size_2_short]
            twords2 = feature_utils.get_words(title2)[: self.matrix_size_2_short]

        matrix = -np.ones((self.matrix_size_2_short, self.matrix_size_2_short))
        for i, word1 in enumerate(twords1):
            for j, word2 in enumerate(twords2):
                v = -1
                if word1 == word2:
                    v = 1
                elif self.use_emb:
                    v = cosine_similarity(self.pretrain_emb[word1].reshape(1, -1),
                                          self.pretrain_emb[word2].reshape(1, -1))[0][0]
                matrix[i][j] = v
        return matrix


class VenueRNNMatchDataset(Dataset):

    def __init__(self, file_dir, max_seq1_len, max_seq2_len, shuffle, seed, args):

        self.max_seq1_len = max_seq1_len
        self.max_seq2_len = max_seq2_len
        self.train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train_filter.txt'), 'r'))

        n_pos_set = int((args.train_num + 2 * args.test_num)/2)

        neg_pairs = [p for p in self.train_data if p[0] == 0]
        pos_pairs = [p for p in self.train_data if p[0] == 1][-n_pos_set:]
        n_pos = len(pos_pairs)
        print("n_pos", n_pos)
        neg_pairs = neg_pairs[-n_pos:]
        self.train_data = pos_pairs + neg_pairs

        self.train_data = sklearn.utils.shuffle(self.train_data, random_state=37)

        t = data_utils.load_large_obj(settings.OUT_DIR, "tokenizer_all_domain.pkl")

        self.vocab_size = len(t.word_counts)
        print("vocab size", self.vocab_size)
        self.load_stop_words()  #TODO

        self.mag = t.texts_to_sequences([p[1] for p in self.train_data])
        self.aminer = t.texts_to_sequences([p[2] for p in self.train_data])
        self.labels = [p[0] for p in self.train_data]
        mag_sum_before = sum([len(x[1].split()) for x in self.train_data])
        mag_sum_after = sum([len(x) for x in self.mag])
        print(mag_sum_before, mag_sum_after, mag_sum_before-mag_sum_after)

        self.calc_keyword_seqs()
        self.mag = pad_sequences(self.mag, maxlen=self.max_seq1_len)
        self.aminer = pad_sequences(self.aminer, maxlen=self.max_seq1_len)
        self.mag_venue_keywords = pad_sequences(self.mag_venue_keywords, maxlen=self.max_seq2_len)
        self.aminer_venue_keywords = pad_sequences(self.aminer_venue_keywords, maxlen=max_seq2_len)

        self.N = len(self.labels)

        n_train = args.train_num
        n_test = args.test_num
        # n_train = self.N - 2*n_test

        train_data = {}
        train_data["x1_seq1"] = self.mag[:n_train]
        train_data["x1_seq2"] = self.mag_venue_keywords[:n_train]
        train_data["x2_seq1"] = self.aminer[:n_train]
        train_data["x2_seq2"] = self.aminer_venue_keywords[:n_train]
        train_data["y"] = self.labels[:n_train]
        train_data["vocab_size"] = self.vocab_size
        print("train labels", len(train_data["y"]))

        test_data = {}
        test_data["x1_seq1"] = self.mag[n_train:(n_train+n_test)]
        test_data["x1_seq2"] = self.mag_venue_keywords[n_train:(n_train+n_test)]
        test_data["x2_seq1"] = self.aminer[n_train:(n_train+n_test)]
        test_data["x2_seq2"] = self.aminer_venue_keywords[n_train:(n_train+n_test)]
        test_data["y"] = self.labels[n_train:(n_train+n_test)]
        print("test labels", len(test_data["y"]))

        valid_data = {}
        valid_data["x1_seq1"] = self.mag[n_train+n_test:(n_train+n_test*2)]
        valid_data["x1_seq2"] = self.mag_venue_keywords[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2_seq1"] = self.aminer[n_train+n_test:(n_train+n_test*2)]
        valid_data["x2_seq2"] = self.aminer_venue_keywords[n_train+n_test:(n_train+n_test*2)]
        valid_data["y"] = self.labels[n_train+n_test:(n_train+n_test*2)]
        print("valid labels", len(valid_data["y"]))

        out_dir = join(settings.DATA_DIR, "dom-adpt")
        os.makedirs(out_dir, exist_ok=True)
        data_utils.dump_large_obj(train_data, out_dir, "venue_rnn_train.pkl")
        data_utils.dump_large_obj(test_data, out_dir, "venue_rnn_test.pkl")
        data_utils.dump_large_obj(valid_data, out_dir, "venue_rnn_valid.pkl")

    def load_stop_words(self):
        self.stop_list = []
        with codecs.open(join(settings.VENUE_DATA_DIR, 'stoplist.txt'), 'r', 'utf-8') as f:
            for word in f.readlines():
                self.stop_list.append(word[:-1])

    def calc_keyword_seqs(self):
        N = len(self.mag)
        mag_keywords = []
        aminer_keywords = []
        for i in range(N):
            cur_v_mag = self.mag[i]
            cur_v_aminer = self.aminer[i]
            overlap = set(cur_v_mag).intersection(cur_v_aminer)
            new_seq_mag = []
            new_seq_aminer = []
            for w in cur_v_mag:
                if w in overlap:
                    new_seq_mag.append(w)
            for w in cur_v_aminer:
                if w in overlap:
                    new_seq_aminer.append(w)
            mag_keywords.append(new_seq_mag)
            aminer_keywords.append(new_seq_aminer)
        self.mag_venue_keywords = mag_keywords
        self.aminer_venue_keywords = aminer_keywords
        print("mag keywords", self.mag_venue_keywords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-dir', type=str, default=settings.VENUE_DATA_DIR, help="Input file directory")
    parser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
    parser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
    parser.add_argument('--train-num', type=int, default=800, help='Training size.')
    parser.add_argument('--test-num', type=int, default=200, help='Testing size.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
    parser.add_argument('--max-sequence-length', type=int, default=17,
                        help="Max sequence length for raw sequences")
    parser.add_argument('--max-key-sequence-length', type=int, default=8,
                        help="Max key sequence length for key sequences")
    args = parser.parse_args()
    filter_venue_dataset()
    dataset = VenueCNNMatchDataset(args.file_dir, args.matrix_size1, args.matrix_size2, args.seed, shuffle=False, args=args, use_emb=False)
    dataset = VenueRNNMatchDataset(args.file_dir, args.max_sequence_length, args.max_key_sequence_length, shuffle=True, seed=args.seed, args=args)

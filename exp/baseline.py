from os.path import join
import pickle
import json
from collections import Counter
import numpy as np
import sklearn
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from utils import feature_utils
from utils import data_utils
from utils import settings
from utils.address_normalization import addressNormalization


def load_aff_data():
    file_dir = settings.AFF_DATA_DIR
    pos_pairs = data_utils.load_json(file_dir, "label_data_aff_zhoushao.json")[:600]
    pos_pairs = [({"name": p["affiliation"]}, {"DisplayName": p["label"]}) for p in pos_pairs if p["label"] != "[NIF]"]
    neg_pairs = data_utils.load_json(file_dir, 'train_negative_affi_clean.json')[:600]
    neg_pairs = [(p['aminer_affi'], p['mag_affi']) for p in neg_pairs]
    pairs_add = data_utils.load_json(file_dir, "mag_aminer_hard_correct_zfj_copy.json")
    print("add pairs", len(pairs_add))
    pos_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "1"]
    neg_pairs += [(p['aminer_affi'], p['mag_affi']) for p in pairs_add if p["label_zfj"] == "0"]
    pos_pairs = pos_pairs[-len(neg_pairs):]
    labels = [1] * len(pos_pairs) + [0] * len(neg_pairs)
    pairs = pos_pairs + neg_pairs  # label balanced is important
    return pairs, labels


def load_venue_data():
    file_dir = settings.VENUE_DATA_DIR
    train_data = json.load(open(join(settings.VENUE_DATA_DIR, 'train_filter.txt'), 'r'))
    return train_data


def fit_tfidf_for_aff():
    corpus = []
    pairs, _ = load_aff_data()
    print(len(pairs), pairs)
    for p in pairs:
        corpus.append(p[0]["name"].lower())
        corpus.append(p[1]["DisplayName"].lower())
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    out_dir = join(settings.OUT_DIR, "aff")
    with open(join(out_dir, "aff_tfidf.pkl"), "wb") as wf:
        pickle.dump(vectorizer, wf)


def fit_tfidf_for_venue():
    corpus = []
    pairs = load_venue_data()
    print(len(pairs), pairs)
    for p in pairs:
        # print(p)
        corpus.append(p[2].lower())
        corpus.append(p[1].lower())
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    out_dir = join(settings.OUT_DIR, "venue")
    with open(join(out_dir, "venue_tfidf.pkl"), "wb") as wf:
        pickle.dump(vectorizer, wf)


def aff_keyword_method():
    pairs, labels = load_aff_data()
    out_dir = join(settings.OUT_DIR, "aff")
    with open(join(out_dir, "aff_tfidf.pkl"), "rb") as rf:
        vectorizer = pickle.load(rf)
    print(vectorizer.vocabulary_)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    pairs, labels = sklearn.utils.shuffle(
        pairs, labels, random_state=42
    )

    features = []
    for i, p in enumerate(pairs):
        aff1 = p[0]["name"].lower()
        aff2 = p[1]["DisplayName"].lower()
        # aff1_words = aff1.split()
        # aff1_words = aff1.split()
        aff1_words = feature_utils.get_words(aff1)
        aff2_words = feature_utils.get_words(aff2)
        # print(aff1_words, aff2_words)
        intersec = Counter(aff1_words) & Counter(aff2_words)
        and_idf = sum([intersec[w] * idf[vocab[w]] for w in intersec if w in vocab])
        # print(is_idf)
        union = Counter(aff1_words) | Counter(aff2_words)
        # or_idf = sum([union[w] * idf[vocab[w]] for w in union if w in vocab])
        aff1_idf = sum([idf[vocab[w]] for w in aff1_words if w in vocab])
        aff2_idf = sum([idf[vocab[w]] for w in aff2_words if w in vocab])
        cur_jac = and_idf/(aff1_idf+aff2_idf-and_idf)
        # print(and_idf, aff1_idf, aff2_idf, cur_jac, labels[i])
        # print(cur_jac, labels[i])
        features.append(cur_jac)

    n = len(pairs)
    n_train = int(n * 0.6)
    n_valid = int(n * 0.2)
    features_valid = features[n_train: (n_valid + n_train)]
    labels_valid = labels[n_train: (n_valid + n_train)]
    features_test = features[(n_valid + n_train): ]
    labels_test = labels[(n_valid + n_train):]

    precs, recs, thrs = precision_recall_curve(labels_valid, features_valid)
    f1s = 2 * precs * recs / (precs + recs)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    print("best thr", best_thr)

    y_pred = np.zeros_like(labels_test)
    y_pred[features_test > best_thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(labels_test, y_pred, average="binary")
    auc = roc_auc_score(labels_test, features_test)
    print("AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f" %
          (auc, prec, rec, f1))


def venue_keyword_method():
    pairs = load_venue_data()
    out_dir = join(settings.OUT_DIR, "venue")
    with open(join(out_dir, "venue_tfidf.pkl"), "rb") as rf:
        vectorizer = pickle.load(rf)
    print(vectorizer.vocabulary_)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    # pairs = sklearn.utils.shuffle(
    #     pairs, random_state=42
    # )

    train_num = 800
    test_num = 200

    n_pos_set = int((train_num + 2 * test_num) / 2)

    neg_pairs = [p for p in pairs if p[0] == 0]
    pos_pairs = [p for p in pairs if p[0] == 1][-n_pos_set:]
    n_pos = len(pos_pairs)
    neg_pairs = neg_pairs[-n_pos:]
    train_data = pos_pairs + neg_pairs

    train_data = sklearn.utils.shuffle(train_data, random_state=37)

    labels = [x[0] for x in train_data]

    features = []
    for i, p in enumerate(train_data):
        aff1 = p[2].lower()
        aff2 = p[1].lower()
        aff1_words = feature_utils.get_words(aff1)
        aff2_words = feature_utils.get_words(aff2)
        intersec = Counter(aff1_words) & Counter(aff2_words)
        and_idf = sum([intersec[w] * idf[vocab[w]] for w in intersec if w in vocab])
        aff1_idf = sum([idf[vocab[w]] for w in aff1_words if w in vocab])
        aff2_idf = sum([idf[vocab[w]] for w in aff2_words if w in vocab])
        cur_jac = and_idf/(aff1_idf+aff2_idf-and_idf)
        features.append(cur_jac)

    n = len(pairs)
    n_train = train_num
    n_valid = test_num
    features_valid = features[n_train: (n_valid + n_train)]
    labels_valid = labels[n_train: (n_valid + n_train)]
    features_test = features[(n_valid + n_train): ]
    labels_test = labels[(n_valid + n_train):]

    print("valid", len(labels_valid),  "test", len(labels_test))

    precs, recs, thrs = precision_recall_curve(labels_valid, features_valid)
    f1s = 2 * precs * recs / (precs + recs)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    print("best thr", best_thr)

    y_pred = np.zeros_like(labels_test)
    y_pred[features_test > best_thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(labels_test, y_pred, average="binary")
    auc = roc_auc_score(labels_test, features_test)
    print("AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f" %
          (auc, prec, rec, f1))


def aff_svm():
    pairs, labels = load_aff_data()
    out_dir = join(settings.OUT_DIR, "aff")
    with open(join(out_dir, "aff_tfidf.pkl"), "rb") as rf:
        vectorizer = pickle.load(rf)
    print(vectorizer.vocabulary_)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    pairs, labels = sklearn.utils.shuffle(
        pairs, labels, random_state=42
    )

    features = []
    for i, p in enumerate(pairs):
        # cur_feat = []
        aff1 = p[0]["name"].lower()
        aff2 = p[1]["DisplayName"].lower()
        # aff1_words = aff1.split()
        # aff1_words = aff1.split()
        aff1_words = feature_utils.get_words(aff1)
        aff2_words = feature_utils.get_words(aff2)
        # print(aff1_words, aff2_words)
        intersec = Counter(aff1_words) & Counter(aff2_words)
        and_idf = sum([intersec[w] * 1 for w in intersec if w in vocab])
        # print(is_idf)
        union = Counter(aff1_words) | Counter(aff2_words)
        # or_idf = sum([union[w] * idf[vocab[w]] for w in union if w in vocab])
        aff1_idf = sum([1 for w in aff1_words if w in vocab])
        aff2_idf = sum([1 for w in aff2_words if w in vocab])
        cur_jac = and_idf/(aff1_idf+aff2_idf-and_idf)

        aff_idf_vec = vectorizer.fit_transform([aff1, aff2])
        # aff2_idf_vec = vectorizer.fit_transform([aff2])
        # print(type(aff_idf_vec), aff_idf_vec)
        # print()
        cos = cosine_similarity(aff_idf_vec)
        # print("cos", cos)
        cur_feat = [cur_jac, cos[0, 1]]
        features.append(cur_feat)

    features = np.array(features)
    n = len(pairs)
    n_train = int(n * 0.6)
    n_valid = int(n * 0.2)
    features_train = features[: n_train + n_valid]
    labels_train = labels[: n_train + n_valid]
    features_test = features[(n_valid + n_train): ]
    labels_test = labels[(n_valid + n_train):]
    clf = svm.SVC()
    clf.fit(features_train, labels_train)
    y_pred = clf.predict(features_test)

    prec, rec, f1, _ = precision_recall_fscore_support(labels_test, y_pred, average="binary")
    print(prec, rec, f1)


def venue_svm():
    pairs = load_venue_data()
    out_dir = join(settings.OUT_DIR, "venue")
    with open(join(out_dir, "venue_tfidf.pkl"), "rb") as rf:
        vectorizer = pickle.load(rf)
    print(vectorizer.vocabulary_)
    vocab = vectorizer.vocabulary_
    idf = vectorizer.idf_

    train_num = 800
    test_num = 200

    n_pos_set = int((train_num + 2 * test_num) / 2)

    neg_pairs = [p for p in pairs if p[0] == 0]
    pos_pairs = [p for p in pairs if p[0] == 1][-n_pos_set:]
    n_pos = len(pos_pairs)
    neg_pairs = neg_pairs[-n_pos:]
    train_data = pos_pairs + neg_pairs

    train_data = sklearn.utils.shuffle(train_data, random_state=37)

    labels = [x[0] for x in train_data]

    features = []
    for i, p in enumerate(train_data):
        aff1 = p[2].lower()
        aff2 = p[1].lower()
        aff1_words = feature_utils.get_words(aff1)
        aff2_words = feature_utils.get_words(aff2)
        intersec = Counter(aff1_words) & Counter(aff2_words)
        and_idf = sum([intersec[w] * 1 for w in intersec if w in vocab])
        aff1_idf = sum([1 for w in aff1_words if w in vocab])
        aff2_idf = sum([1 for w in aff2_words if w in vocab])
        cur_jac = and_idf/(aff1_idf+aff2_idf-and_idf)

        aff_idf_vec = vectorizer.fit_transform([aff1, aff2])
        cos = cosine_similarity(aff_idf_vec)

        cur_v_mag = aff1
        cur_v_aminer = aff2
        overlap = set(cur_v_mag).intersection(cur_v_aminer)
        new_seq_mag = []
        new_seq_aminer = []
        for w in cur_v_mag:
            if w in overlap:
                new_seq_mag.append(w)
        for w in cur_v_aminer:
            if w in overlap:
                new_seq_aminer.append(w)

        intersec = Counter(new_seq_aminer) & Counter(new_seq_mag)
        and_idf = sum([intersec[w] * 1 for w in intersec if w in vocab])
        aminer_idf_key = sum([1 for w in new_seq_aminer if w in vocab])
        mag_idf_key = sum([1 for w in new_seq_mag if w in vocab])
        mother = aminer_idf_key+mag_idf_key-and_idf
        if mother != 0:
            cur_jac_key = and_idf/mother
            print("key jac", cur_jac_key)
            aff_idf_vec = vectorizer.fit_transform([" ".join(new_seq_aminer), " ".join(new_seq_mag)])
            cos_key = cosine_similarity(aff_idf_vec)
        else:
            cur_jac_key = 0
            cos_key = -1
            # print("here")

        # cur_feat = [cur_jac, cos[0, 1], cur_jac_key, cos_key]
        cur_feat = [cur_jac, cos[0, 1]]
        # print("cur feature", cur_feat)
        features.append(cur_feat)

    features = np.array(features)
    # n = len(pairs)
    n_train = train_num
    n_valid = test_num
    features_train = features[: n_train + n_valid]
    labels_train = labels[: n_train + n_valid]
    features_test = features[(n_valid + n_train): ]
    labels_test = labels[(n_valid + n_train):]
    print("n_test", len(labels_test))
    clf = svm.SVC()
    clf.fit(features_train, labels_train)
    y_pred = clf.predict(features_test)

    prec, rec, f1, _ = precision_recall_fscore_support(labels_test, y_pred, average="binary")
    print(prec, rec, f1)


def gen_aff_record_linkage_table():
    pairs, labels = load_aff_data()
    pairs, labels = sklearn.utils.shuffle(
        pairs, labels, random_state=42
    )
    n = len(pairs)
    n_train = int(n * 0.6)
    n_valid = int(n * 0.2)
    aff_to_aid = {}
    cur_idx = 0
    # table1_aff = []
    # table2_aff = []
    out_dir = join(settings.OUT_DIR, "aff")
    wf1 = open(join(out_dir, "aff_train1.csv"), "w")
    wf2 = open(join(out_dir, "aff_train2.csv"), "w")
    wf1.write("name,main_body,uid\n")
    wf2.write("name,main_body,uid\n")
    test_pairs = []
    valid_pairs = []
    neg_cnt = 0
    an = addressNormalization()
    for i, p in enumerate(pairs):
        aff1_short = an.find_inst(p[0]["name"])[1].lower().replace(",", " ")
        aff1 = p[0]["name"].lower().replace(",", " ")
        # aff2_short = an.find_inst(p[1]["DisplayName"])[1].lower()
        aff2 = p[1]["DisplayName"].lower().replace(",", " ")
        label = labels[i]
        # if aff2 in aff_to_aid:
        #     continue
        if label == 1:
            aff_to_aid[aff2] = cur_idx
            aff_to_aid[aff1] = cur_idx
            cur_idx += 1
        else:
            aff_to_aid[aff2] = cur_idx
            aff_to_aid[aff1] = cur_idx + 1
            cur_idx += 2
        if i < n_train:
            wf1.write(aff1 + "," + aff1_short + "," + str(aff_to_aid[aff1]) + "\n")
            wf2.write(aff2 + "," + aff2 + "," + str(aff_to_aid[aff2]) + "\n")
        elif i < n_train + n_valid:
            valid_pairs.append(
                ({"name": aff1, "main_body": aff1_short, "uid": str(aff_to_aid[aff1])},
                 {"name": aff2, "main_body": aff2, "uid": str(aff_to_aid[aff2])})
            )
        else:
            test_pairs.append(
                ({"name": aff1, "main_body": aff1_short, "uid": str(aff_to_aid[aff1])},
                 {"name": aff2, "main_body": aff2, "uid": str(aff_to_aid[aff2])})
            )
            if aff_to_aid[aff1] != aff_to_aid[aff2]:
                neg_cnt += 1
    wf1.close()
    wf2.close()

    print(len(test_pairs), neg_cnt)

    data_utils.dump_json(test_pairs, out_dir, "valid_aff_dedupe_pairs.json")
    data_utils.dump_json(test_pairs, out_dir, "test_aff_dedupe_pairs.json")


def gen_venue_record_linkage_table():
    pairs = load_venue_data()
    train_num = 800
    test_num = 200

    n_pos_set = int((train_num + 2 * test_num) / 2)

    neg_pairs = [p for p in pairs if p[0] == 0]
    pos_pairs = [p for p in pairs if p[0] == 1][-n_pos_set:]
    n_pos = len(pos_pairs)
    neg_pairs = neg_pairs[-n_pos:]
    train_data = pos_pairs + neg_pairs

    train_data = sklearn.utils.shuffle(train_data, random_state=37)

    labels = [x[0] for x in train_data]

    # n = len(pairs)
    n_train = train_num
    n_valid = test_num
    aff_to_aid = {}
    cur_idx = 0
    # table1_aff = []
    # table2_aff = []
    out_dir = join(settings.OUT_DIR, "venue")
    wf1 = open(join(out_dir, "venue_train1.csv"), "w")
    wf2 = open(join(out_dir, "venue_train2.csv"), "w")
    wf1.write("name,main_body,uid\n")
    wf2.write("name,main_body,uid\n")
    test_pairs = []
    valid_pairs = []
    neg_cnt = 0
    # an = addressNormalization()
    for i, p in enumerate(train_data):
        # aff1_short = an.find_inst(p[0]["name"])[1].lower().replace(",", " ")
        aff1 = p[2].lower().replace(",", " ")
        # aff2_short = an.find_inst(p[1]["DisplayName"])[1].lower()
        aff2 = p[1].lower().replace(",", " ")
        label = labels[i]
        # if aff2 in aff_to_aid:
        #     continue
        if label == 1:
            aff_to_aid[aff2] = cur_idx
            aff_to_aid[aff1] = cur_idx
            cur_idx += 1
        else:
            aff_to_aid[aff2] = cur_idx
            aff_to_aid[aff1] = cur_idx + 1
            cur_idx += 2

        cur_v_mag = aff1.split()
        cur_v_aminer = aff2.split()
        overlap = set(cur_v_mag).intersection(cur_v_aminer)
        new_seq_mag = []
        new_seq_aminer = []
        for w in cur_v_mag:
            if w in overlap:
                new_seq_mag.append(w)
        for w in cur_v_aminer:
            if w in overlap:
                new_seq_aminer.append(w)

        if i < n_train:
            wf1.write(aff1 + "," + " ".join(new_seq_mag) + "," + str(aff_to_aid[aff1]) + "\n")
            wf2.write(aff2 + "," + " ".join(new_seq_aminer) + "," + str(aff_to_aid[aff2]) + "\n")
        elif i < n_train + n_valid:
            valid_pairs.append(
                ({"name": aff1, "main_body": " ".join(new_seq_mag), "uid": str(aff_to_aid[aff1])},
                 {"name": aff2, "main_body": " ".join(new_seq_aminer), "uid": str(aff_to_aid[aff2])})
            )
        else:
            test_pairs.append(
                ({"name": aff1, "main_body": " ".join(new_seq_mag), "uid": str(aff_to_aid[aff1])},
                 {"name": aff2, "main_body": " ".join(new_seq_aminer), "uid": str(aff_to_aid[aff2])})
            )
            if aff_to_aid[aff1] != aff_to_aid[aff2]:
                neg_cnt += 1
    wf1.close()
    wf2.close()

    print(len(test_pairs), neg_cnt)

    data_utils.dump_json(test_pairs, out_dir, "valid_venue_dedupe_pairs.json")
    data_utils.dump_json(test_pairs, out_dir, "test_venue_dedupe_pairs.json")


if __name__ == "__main__":
    # pairs, _ = load_aff_data()
    # fit_tfidf_for_aff()
    # aff_keyword_method()
    # aff_svm()
    gen_aff_record_linkage_table()
    # fit_tfidf_for_venue()
    # venue_keyword_method()
    # venue_svm()
    # gen_venue_record_linkage_table()

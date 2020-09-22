import os
import numpy as np

from utils import settings


def gen_mean_tranfer_results(dst, n_sources, attn, base_model):
    model_dir = os.path.join(settings.OUT_DIR, dst)
    with open(os.path.join(model_dir, "{}_{}_moe_attn_{}_sources_{}_train_num_{}_results.txt".format(
        dst, base_model, attn, n_sources, None)), "r") as rf:

        val_losses = []
        auc_list = []
        prec_list = []
        rec_list = []
        f1_list = []
        val_weight_list = []
        test_weight_list = []

        for i, line in enumerate(rf):
            line = line.strip()
            if line.startswith("min"):
                items = line.split(",")
                n_start_val_loss = len("min valid loss ")
                val_loss = float(items[0][n_start_val_loss:])
                cur_auc = float(items[1].split(":")[-1][1:])
                cur_prec = float(items[2].split(":")[-1][1:])
                cur_rec = float(items[3].split(":")[-1][1:])
                cur_f1 = float(items[4].split(":")[-1][1:])
                val_losses.append(val_loss)
                auc_list.append(cur_auc)
                prec_list.append(cur_prec)
                rec_list.append(cur_rec)
            elif line.startswith("val"):
                n_start_val_weights = len("val weights:")
                items = line[n_start_val_weights:].split(",")
                cur_weights = []
                for s in range(n_sources):
                    cur_w = float(items[s][1:])
                    cur_weights.append(cur_w)
                val_weight_list.append(cur_weights)
            elif line.startswith("test"):
                n_start_test_weights = len("test weights:")
                items = line[n_start_test_weights:].split(",")
                cur_weights = []
                for s in range(n_sources):
                    cur_w = float(items[s][1:])
                    cur_weights.append(cur_w)
                test_weight_list.append(cur_weights)
    val_losses = np.array(val_losses)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    auc_list = np.array(auc_list)

    val_weight_list = np.array(val_weight_list)
    test_weight_list = np.array(test_weight_list)

    assert len(val_weight_list) == 10

    val_loss_mean = np.mean(val_losses)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)
    f1_mean = 2 * prec_mean * rec_mean / (prec_mean + rec_mean)
    auc_mean = np.mean(auc_list)

    val_weight_mean = np.mean(val_weight_list, axis=0)
    val_weight_std = np.std(val_weight_list, axis=0)
    test_weight_mean = np.mean(test_weight_list, axis=0)
    test_weight_std = np.std(test_weight_list, axis=0)

    with open(os.path.join(model_dir, "{}_{}_attn_{}_sources_{}_avg_results.txt".format(
        dst, base_model, attn, n_sources
    )), "w") as wf:
        wf.write("val loss mean: {:.4f}\n".format(val_loss_mean))
        wf.write("test results: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            auc_mean, prec_mean, rec_mean, f1_mean
        ))
        wf.write("val weights: ")
        for s in range(n_sources):
            wf.write("{:.4f} ({:.4f}),".format(val_weight_mean[s], val_weight_std[s]))
        wf.write("\n")

        wf.write("test weights: ")
        for s in range(n_sources):
            wf.write("{:.4f} ({:.4f}),".format(test_weight_mean[s], test_weight_std[s]))
        wf.write("\n")


def gen_mean_train_results(dst, base_model):
    model_dir = os.path.join(settings.OUT_DIR, dst)
    with open(os.path.join(model_dir, "{}_{}_train_num_{}_results.txt".format(
        dst, base_model, None)), "r") as rf:

        val_losses = []
        auc_list = []
        prec_list = []
        rec_list = []

        for i, line in enumerate(rf):
            line = line.strip()
            if line.startswith("min"):
                items = line.split(",")
                n_start_val_loss = len("min valid loss ")
                val_loss = float(items[0][n_start_val_loss:])
                cur_auc = float(items[1].split(":")[-1][1:])
                cur_prec = float(items[2].split(":")[-1][1:])
                cur_rec = float(items[3].split(":")[-1][1:])
                cur_f1 = float(items[4].split(":")[-1][1:])
                val_losses.append(val_loss)
                auc_list.append(cur_auc)
                prec_list.append(cur_prec)
                rec_list.append(cur_rec)
    val_losses = np.array(val_losses)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    auc_list = np.array(auc_list)

    assert len(val_losses) == 10

    val_loss_mean = np.mean(val_losses)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)
    f1_mean = 2 * prec_mean * rec_mean / (prec_mean + rec_mean)
    auc_mean = np.mean(auc_list)

    with open(os.path.join(model_dir, "{}_{}_train_avg_results.txt".format(
        dst, base_model)), "w") as wf:
        wf.write("val loss mean: {:.4f}\n".format(val_loss_mean))
        wf.write("test results: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            auc_mean, prec_mean, rec_mean, f1_mean
        ))


def gen_all_avg_results():
    # dst, n_sources, attn, base_model
    dst_range = ["aff", "author", "paper", "venue"]
    sources = [3, 4]
    attn_types = ["onehot", "mlp"]
    base_models = ["cnn", "rnn"]
    for dst in dst_range:
        for n_src in sources:
            for attn in attn_types:
                for bm in base_models:
                    gen_mean_tranfer_results(dst, n_src, attn, bm)


def gen_single_train_results():
    dst_range = ["aff", "author", "paper", "venue"]
    base_models = ["cnn", "rnn"]
    for dst in dst_range:
        for bm in base_models:
            gen_mean_train_results(dst, bm)


def gen_single_trans_last_3_mean_for_one(dst, base_model, idx):
    model_dir = os.path.join(settings.OUT_DIR, dst)
    with open(os.path.join(model_dir, "{}_{}_single_domain_trans_train_num_{}_results.txt".format(
        dst, base_model, None)), "r") as rf:

        val_losses = []
        auc_list = []
        prec_list = []
        rec_list = []
        src_list = []
        n_hit = 0

        for i, line in enumerate(rf):
            line = line.strip()
            if "avg min" in line:
                items = line.split(",")
                if n_hit == idx:
                    n_hit += 1
                    continue
                n_hit += 1
                cur_src = items[0].split()[-1]
                n_start_val_loss = len(" avg min valid loss ")
                val_loss = float(items[1][n_start_val_loss:])
                cur_auc = float(items[2].split(":")[-1][1:])
                cur_prec = float(items[3].split(":")[-1][1:])
                cur_rec = float(items[4].split(":")[-1][1:])
                cur_f1 = float(items[5].split(":")[-1][1:])
                val_losses.append(val_loss)
                auc_list.append(cur_auc)
                prec_list.append(cur_prec)
                rec_list.append(cur_rec)
                src_list.append(cur_src)
    val_losses = np.array(val_losses)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    auc_list = np.array(auc_list)

    assert len(val_losses) == 3

    val_loss_mean = np.mean(val_losses)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)
    f1_mean = 2 * prec_mean * rec_mean / (prec_mean + rec_mean)
    auc_mean = np.mean(auc_list)

    with open(os.path.join(model_dir, "{}_{}_single_trans_last_3_avg_results.txt".format(
        dst, base_model)), "w") as wf:
        wf.write("val loss mean: {:.4f}\n".format(val_loss_mean))
        wf.write("test results: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            auc_mean, prec_mean, rec_mean, f1_mean
        ))


def gen_single_trans_last_1_mean_for_one(dst, base_model, idx):
    model_dir = os.path.join(settings.OUT_DIR, dst)
    with open(os.path.join(model_dir, "{}_{}_single_domain_trans_train_num_{}_tune_1_results.txt".format(
        dst, base_model, None)), "r") as rf:

        val_losses = []
        auc_list = []
        prec_list = []
        rec_list = []
        src_list = []
        n_hit = 0

        for i, line in enumerate(rf):
            line = line.strip()
            if "avg min" in line:
                items = line.split(",")
                if n_hit == idx:
                    n_hit += 1
                    continue
                n_hit += 1
                cur_src = items[0].split()[-1]
                n_start_val_loss = len(" avg min valid loss ")
                val_loss = float(items[1][n_start_val_loss:])
                cur_auc = float(items[2].split(":")[-1][1:])
                cur_prec = float(items[3].split(":")[-1][1:])
                cur_rec = float(items[4].split(":")[-1][1:])
                cur_f1 = float(items[5].split(":")[-1][1:])
                val_losses.append(val_loss)
                auc_list.append(cur_auc)
                prec_list.append(cur_prec)
                rec_list.append(cur_rec)
                src_list.append(cur_src)
    val_losses = np.array(val_losses)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    auc_list = np.array(auc_list)

    assert len(val_losses) == 3

    val_loss_mean = np.mean(val_losses)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)
    f1_mean = 2 * prec_mean * rec_mean / (prec_mean + rec_mean)
    auc_mean = np.mean(auc_list)

    with open(os.path.join(model_dir, "{}_{}_single_trans_last_1_avg_results.txt".format(
        dst, base_model)), "w") as wf:
        wf.write("val loss mean: {:.4f}\n".format(val_loss_mean))
        wf.write("test results: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            auc_mean, prec_mean, rec_mean, f1_mean
        ))


def gen_single_trans_direct_for_one(dst, base_model, idx):
    model_dir = os.path.join(settings.OUT_DIR, dst)
    with open(os.path.join(model_dir, "{}_{}_direct_transfer_results.txt".format(
        dst, base_model)), "r") as rf:

        val_losses = []
        auc_list = []
        prec_list = []
        rec_list = []
        src_list = []
        n_hit = 0

        for i, line in enumerate(rf):
            line = line.strip()
            if "avg min" in line:
                items = line.split(",")
                if n_hit == idx:
                    n_hit += 1
                    continue
                n_hit += 1
                cur_src = items[0].split()[-1]
                n_start_val_loss = len(" avg min valid loss ")
                val_loss = float(items[1][n_start_val_loss:])
                cur_auc = float(items[2].split(":")[-1][1:])
                cur_prec = float(items[3].split(":")[-1][1:])
                cur_rec = float(items[4].split(":")[-1][1:])
                cur_f1 = float(items[5].split(":")[-1][1:])
                val_losses.append(val_loss)
                auc_list.append(cur_auc)
                prec_list.append(cur_prec)
                rec_list.append(cur_rec)
                src_list.append(cur_src)
    val_losses = np.array(val_losses)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    auc_list = np.array(auc_list)

    assert len(val_losses) == 3

    val_loss_mean = np.mean(val_losses)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)
    f1_mean = 2 * prec_mean * rec_mean / (prec_mean + rec_mean)
    auc_mean = np.mean(auc_list)

    with open(os.path.join(model_dir, "{}_{}_single_trans_direct_avg_results.txt".format(
        dst, base_model)), "w") as wf:
        wf.write("val loss mean: {:.4f}\n".format(val_loss_mean))
        wf.write("test results: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            auc_mean, prec_mean, rec_mean, f1_mean
        ))


def gen_single_train_rnn_for_one(dst, n_seq):
    model_dir = os.path.join(settings.OUT_DIR, dst)
    with open(os.path.join(model_dir, "{}_rnn_train_num_None_results_n_seq_{}.txt".format(
        dst,  n_seq)), "r") as rf:

        val_losses = []
        auc_list = []
        prec_list = []
        rec_list = []

        for i, line in enumerate(rf):
            line = line.strip()
            if line.startswith("min"):
                items = line.split(",")
                n_start_val_loss = len("min valid loss ")
                val_loss = float(items[0][n_start_val_loss:])
                cur_auc = float(items[1].split(":")[-1][1:])
                cur_prec = float(items[2].split(":")[-1][1:])
                cur_rec = float(items[3].split(":")[-1][1:])
                cur_f1 = float(items[4].split(":")[-1][1:])
                val_losses.append(val_loss)
                auc_list.append(cur_auc)
                prec_list.append(cur_prec)
                rec_list.append(cur_rec)
    val_losses = np.array(val_losses)
    prec_list = np.array(prec_list)
    rec_list = np.array(rec_list)
    auc_list = np.array(auc_list)

    assert len(val_losses) == 10

    val_loss_mean = np.mean(val_losses)
    prec_mean = np.mean(prec_list)
    rec_mean = np.mean(rec_list)
    f1_mean = 2 * prec_mean * rec_mean / (prec_mean + rec_mean)
    auc_mean = np.mean(auc_list)

    with open(os.path.join(model_dir, "{}_rnn_train_n_seq_{}_avg_results.txt".format(
        dst, n_seq)), "w") as wf:
        wf.write("val loss mean: {:.4f}\n".format(val_loss_mean))
        wf.write("test results: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            auc_mean, prec_mean, rec_mean, f1_mean
        ))


def gen_single_trans_last_3_avg_results():
    dst_range = ["aff", "author", "paper", "venue"]
    base_models = ["cnn", "rnn"]

    for i in range(len(dst_range)):
        for bm in base_models:
            gen_single_trans_last_3_mean_for_one(dst_range[i], bm, i)


def gen_single_trans_last_1_avg_results():
    dst_range = ["aff", "author", "paper", "venue"]
    base_models = ["cnn", "rnn"]

    for i in range(len(dst_range)):
        for bm in base_models:
            gen_single_trans_last_1_mean_for_one(dst_range[i], bm, i)


def gen_direct_trans_avg_results():
    dst_range = ["aff", "author", "paper", "venue"]
    base_models = ["cnn", "rnn"]

    for i in range(len(dst_range)):
        for bm in base_models:
            gen_single_trans_direct_for_one(dst_range[i], bm, i)


def gen_ab_study_avg_results():
    n_seqs = [1, 2]
    dst_range = ["aff"]
    for dst in dst_range:
        for n_seq in n_seqs:
            gen_single_train_rnn_for_one(dst, n_seq)


if __name__ == "__main__":
    # gen_all_avg_results()
    # gen_single_train_results()
    # gen_single_trans_last_3_avg_results()
    # gen_single_trans_last_1_avg_results()
    # gen_direct_trans_avg_results()
    gen_ab_study_avg_results()

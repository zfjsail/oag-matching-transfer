import os
import shutil
import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from utils.data_utils import say, softmax
from dataset import ProcessedCNNInputDataset, ProcessedRNNInputDataset
from models.cnn import CNNMatchModel
from models.rnn import BiLSTM
from models.attn import MulInteractAttention, OneHotAttention, MLP

from utils import settings

import warnings

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser(description="Learning to Adapt from Single Source Domain")
argparser.add_argument("--train", type=str, default="aff,author,paper,venue",
                       help="multi-source domains for training, separated with (,)")
argparser.add_argument("--test", type=str, default="aff",
                       help="target domain for testing")
argparser.add_argument("--eval_only", action="store_true")
argparser.add_argument("--batch_size", type=int, default=32)
argparser.add_argument("--max_epoch", type=int, default=300)
argparser.add_argument("--lr", type=float, default=1e-1)
argparser.add_argument("--lambda_entropy", type=float, default=0.0)
argparser.add_argument("--lambda_moe", type=float, default=1)
argparser.add_argument("--base_model", type=str, default="rnn")
argparser.add_argument("--attn-type", type=str, default="mlp")
argparser.add_argument('--train-num', default=None, help='Number of training samples')
argparser.add_argument('--n-try', type=int, default=5, help='Repeat Times')

argparser.add_argument('--embedding-size', type=int, default=128,
                       help="Embeding size for LSTM layer")
argparser.add_argument('--hidden-size', type=int, default=32,
                       help="Hidden size for LSTM layer")
argparser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate (1 - keep probability).')
argparser.add_argument('--max-vocab-size', type=int, default=settings.MAX_WORD_TOKEN_NUM, help="Maximum of Vocab Size")
argparser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
argparser.add_argument('--matrix-size1', type=int, default=7, help='Matrix size 1.')
argparser.add_argument('--matrix-size2', type=int, default=4, help='Matrix size 2.')
argparser.add_argument('--mat1-channel1', type=int, default=4, help='Matrix1 number of channels1.')
argparser.add_argument('--mat1-kernel-size1', type=int, default=3, help='Matrix1 kernel size1.')
argparser.add_argument('--mat1-channel2', type=int, default=8, help='Matrix1 number of channel2.')
argparser.add_argument('--mat1-kernel-size2', type=int, default=2, help='Matrix1 kernel size2.')
argparser.add_argument('--mat1-hidden', type=int, default=64, help='Matrix1 hidden dim.')
argparser.add_argument('--mat2-channel1', type=int, default=4, help='Matrix2 number of channels1.')
argparser.add_argument('--mat2-kernel-size1', type=int, default=2, help='Matrix2 kernel size1.')
argparser.add_argument('--mat2-hidden', type=int, default=64, help='Matrix2 hidden dim')
argparser.add_argument('--build-index-window', type=int, default=5, help='Matrix2 hidden dim')
argparser.add_argument('--seed', type=int, default=42, help='Random seed.')
argparser.add_argument('--seed-delta', type=int, default=0, help='Random seed.')

argparser.add_argument('--weight-decay', type=float, default=1e-3,
                       help='Weight decay (L2 loss on parameters).')
argparser.add_argument('--check-point', type=int, default=2, help="Check point")
argparser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")

args, _ = argparser.parse_known_args()

n_sources = len(args.train.split(','))
sources_all = ["aff", "author", "paper", "venue"]
source_to_idx = {sources_all[x]: x for x in range(len(sources_all))}
print("source to idx", source_to_idx)


def evaluate(epoch, encoders, classifier, data_loader, return_best_thrs, args, writer, thr=None):
    encoder_src, encoder_dst = encoders
    map(lambda m: m.eval(), [encoder_src, classifier, encoder_dst])
    correct = 0
    tot_cnt = 0
    y_true = []
    y_pred = []
    y_score = []
    loss = 0.

    if args.base_model == "cnn":
        for batch1, batch2, label in data_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                label = label.cuda()

            batch1 = Variable(batch1)
            batch2 = Variable(batch2)
            bs = len(batch1)

            _, cur_hidden = encoder_src(batch1, batch2)
            cur_output = classifier(cur_hidden)

            output = F.softmax(cur_output, dim=1)

            pred = output.data.max(dim=1)[1]

            loss_batch = F.nll_loss(torch.log(output), label)
            loss += bs * loss_batch.item()

            y_true += label.tolist()
            y_pred += pred.tolist()
            correct += pred.eq(label).sum()
            tot_cnt += output.size(0)
            y_score += output[:, 1].data.tolist()
    elif args.base_model == "rnn":
        for batch1, batch2, batch3, batch4, label in data_loader:
            if args.cuda:
                batch1 = batch1.cuda()
                batch2 = batch2.cuda()
                batch3 = batch3.cuda()
                batch4 = batch4.cuda()
                label = label.cuda()

            # batch1 = Variable(batch1)
            # batch2 = Variable(batch2)
            bs = len(batch1)

            _, cur_hidden = encoder_src(batch1, batch2, batch3, batch4)
            cur_output = classifier(cur_hidden)

            output = F.softmax(cur_output, dim=1)

            pred = output.data.max(dim=1)[1]

            loss_batch = F.nll_loss(torch.log(output), label)
            loss += bs * loss_batch.item()

            y_true += label.tolist()
            y_pred += pred.tolist()
            correct += pred.eq(label).sum()
            tot_cnt += output.size(0)
            y_score += output[:, 1].data.tolist()
    else:
        raise NotImplementedError

    if thr is not None:
        print("using threshold %.4f" % thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    loss /= tot_cnt

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    print("Loss: {:.4f}, AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}".format(
        loss, auc * 100, prec * 100, rec * 100, f1 * 100))

    best_thr = None
    metric = [loss, auc, prec, rec, f1]

    if return_best_thrs:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        print("best threshold={:.4f}, f1={:.4f}".format(best_thr, np.max(f1s)))

        writer.add_scalar('val_loss',
                          loss,
                          epoch)
    else:
        writer.add_scalar('test_f1',
                          f1,
                          epoch)

    return best_thr, metric


def train_epoch(iter_cnt, encoders, classifier, train_loader_dst, args, optim_model, epoch, writer):
    encoder_src, encoder_dst = encoders

    map(lambda m: m.train(), [classifier, encoder_src, encoder_dst])
    moe_criterion = nn.NLLLoss()  # with log_softmax separated
    loss_total = 0
    n_batch = 0

    for batch in train_loader_dst:
        if args.base_model == "cnn":
            batch1, batch2, label = batch
        elif args.base_model == "rnn":
            batch1, batch2, batch3, batch4, label = batch
        else:
            raise NotImplementedError

        bs = len(label)

        iter_cnt += 1
        n_batch += 1
        if args.cuda:
            batch1 = batch1.cuda()
            batch2 = batch2.cuda()
            label = label.cuda()
            if args.base_model == "rnn":
                batch3 = batch3.cuda()
                batch4 = batch4.cuda()

        if args.base_model == "cnn":
            _, hidden_from_dst_enc = encoder_dst(batch1, batch2)
        elif args.base_model == "rnn":
            _, hidden_from_dst_enc = encoder_dst(batch1, batch2, batch3, batch4)
        else:
            raise NotImplementedError

        if args.base_model == "cnn":
            _, cur_hidden = encoder_src(batch1, batch2)
        elif args.base_model == "rnn":
            _, cur_hidden = encoder_src(batch1, batch2, batch3, batch4)
        else:
            raise NotImplementedError
        cur_output = classifier(cur_hidden)

        optim_model.zero_grad()

        output = F.softmax(cur_output, dim=1)
        loss_moe = moe_criterion(torch.log(output), label)
        loss = args.lambda_moe * loss_moe

        loss_total += loss.item()
        loss.backward()
        optim_model.step()

        if iter_cnt % 10 == 0:
            say("{} MOE loss: {:.4f}, "
                "loss: {:.4f}\n"
                .format(iter_cnt,
                        loss_moe.item(),
                        loss.data.item()
                        ))

    loss_total /= n_batch
    writer.add_scalar('training_loss',
                      loss_total,
                      epoch)

    say("\n")
    return iter_cnt


def train_single_domain_transfer(args, wf, src, repeat_seed):
    tb_dir = 'runs/{}_sup_base_{}_source_{}_train_num_{}_{}'.format(
        args.test, args.base_model, src, args.train_num, repeat_seed)
    if os.path.exists(tb_dir) and os.path.isdir(tb_dir):
        shutil.rmtree(tb_dir)
    writer = SummaryWriter(tb_dir)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    say('cuda is available %s\n' % args.cuda)

    np.random.seed(args.seed + repeat_seed)
    torch.manual_seed(args.seed + repeat_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + repeat_seed)

    pretrain_emb = torch.load(os.path.join(settings.OUT_DIR, "rnn_init_word_emb.emb"))

    src_model_dir = os.path.join(settings.OUT_DIR, src)

    if args.base_model == "cnn":
        encoder_src = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                      mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                      mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                      mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                      mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    elif args.base_model == "rnn":
        encoder_src = BiLSTM(pretrain_emb=pretrain_emb,
                                   vocab_size=args.max_vocab_size,
                                   embedding_size=args.embedding_size,
                                   hidden_size=args.hidden_size,
                                   dropout=args.dropout)
    else:
        raise NotImplementedError
    if args.cuda:
        encoder_src.load_state_dict(
            torch.load(os.path.join(src_model_dir,
                                    "{}-match-best-now-train-num-{}-try-{}.mdl".format(args.base_model, args.train_num, repeat_seed))))
    else:
        encoder_src.load_state_dict(
            torch.load(os.path.join(src_model_dir,
                                    "{}-match-best-now-train-num-{}-try-{}.mdl".format(args.base_model, args.train_num, repeat_seed)),
                       map_location=torch.device('cpu')))

    dst_model_dir = os.path.join(settings.OUT_DIR, args.test)
    if args.base_model == "cnn":
        encoder_dst_pretrain = CNNMatchModel(input_matrix_size1=args.matrix_size1, input_matrix_size2=args.matrix_size2,
                                             mat1_channel1=args.mat1_channel1, mat1_kernel_size1=args.mat1_kernel_size1,
                                             mat1_channel2=args.mat1_channel2, mat1_kernel_size2=args.mat1_kernel_size2,
                                             mat1_hidden=args.mat1_hidden, mat2_channel1=args.mat2_channel1,
                                             mat2_kernel_size1=args.mat2_kernel_size1, mat2_hidden=args.mat2_hidden)
    elif args.base_model == "rnn":
        encoder_dst_pretrain = BiLSTM(pretrain_emb=pretrain_emb,
                                      vocab_size=args.max_vocab_size,
                                      embedding_size=args.embedding_size,
                                      hidden_size=args.hidden_size,
                                      dropout=args.dropout)
    else:
        raise NotImplementedError

    encoder_dst_pretrain.load_state_dict(
        torch.load(os.path.join(dst_model_dir,
                                "{}-match-best-now-train-num-{}-try-{}.mdl".format(args.base_model, args.train_num, repeat_seed))))

    # args = argparser.parse_args()
    say(args)
    print()

    say("Transferring from %s to %s\n" % (src, args.test))

    if args.base_model == "cnn":
        train_dataset_dst = ProcessedCNNInputDataset(args.test, "train", args.train_num)
        valid_dataset = ProcessedCNNInputDataset(args.test, "valid")
        test_dataset = ProcessedCNNInputDataset(args.test, "test")

    elif args.base_model == "rnn":
        train_dataset_dst = ProcessedRNNInputDataset(args.test, "train", args.train_num)
        valid_dataset = ProcessedRNNInputDataset(args.test, "valid")
        test_dataset = ProcessedRNNInputDataset(args.test, "test")
    else:
        raise NotImplementedError

    print("train num", len(train_dataset_dst))

    train_loader_dst = data.DataLoader(
        train_dataset_dst,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    say("Corpus loaded.\n")

    classifier = nn.Sequential(
        nn.Linear(encoder_src.n_out, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )

    if args.cuda:
        encoder_dst_pretrain.cuda()
        encoder_src.cuda()
        classifier.cuda()

    requires_grad = lambda x: x.requires_grad
    task_params = list(classifier.parameters())

    if args.base_model == "cnn":
        optim_model = optim.Adagrad(
            filter(requires_grad, task_params),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.base_model == "rnn":
        optim_model = optim.Adam(
            filter(requires_grad, task_params),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise NotImplementedError

    say("Training will begin from scratch\n")

    iter_cnt = 0
    min_loss_val = None
    max_auc_val = None
    best_test_results = None
    weights_sources = None
    model_dir = os.path.join(settings.OUT_DIR, args.test)

    cur_src_idx = source_to_idx[src]

    for epoch in range(args.max_epoch):
        print("training epoch", epoch)

        iter_cnt = train_epoch(
            iter_cnt,
            [encoder_src, encoder_dst_pretrain], classifier,
            train_loader_dst,
            args,
            optim_model,
            epoch,
            writer
        )

        thr, metrics_val = evaluate(
            epoch,
            [encoder_src, encoder_dst_pretrain], classifier,
            valid_loader,
            True,
            args,
            writer
        )

        _, metrics_test = evaluate(
            epoch,
            [encoder_src, encoder_dst_pretrain], classifier,
            test_loader,
            False,
            args,
            writer,
            thr=thr
        )

        if min_loss_val is None or min_loss_val > metrics_val[0]:
            print("change val loss from {} to {}".format(min_loss_val, metrics_val[0]))
            min_loss_val = metrics_val[0]
            best_test_results = metrics_test
            torch.save(classifier, os.path.join(model_dir, "{}_{}_classifier_from_src_{}_train_num_{}_try_{}.mdl".format(
                args.test, args.base_model, cur_src_idx, args.train_num, repeat_seed
            )))


    print()
    print("src: {}, min valid loss {:.4f}, best test metrics: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
        src, min_loss_val, best_test_results[1] * 100, best_test_results[2] * 100, best_test_results[3] * 100,
                              best_test_results[4] * 100
            ))

    wf.write(
        "from src {}, min valid loss {:.4f}, best test metrics: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            src, min_loss_val, best_test_results[1] * 100, best_test_results[2] * 100, best_test_results[3] * 100,
                          best_test_results[4] * 100
        ))
    writer.close()

    return min_loss_val, best_test_results[1:]


def train(args):
    model_dir = os.path.join(settings.OUT_DIR, args.test)
    wf = open(os.path.join(model_dir, "{}_{}_single_domain_trans_train_num_{}_results.txt".format(
        args.test, args.base_model, args.train_num)), "w")
    test_idx = source_to_idx[args.test]
    for src in sources_all:
        min_loss_val_list = []
        test_results_list = []
        for t in range(args.n_try):
            cur_min_loss_val, cur_test_results = train_single_domain_transfer(args, wf, src, t)
            min_loss_val_list.append(cur_min_loss_val)
            test_results_list.append(cur_test_results)
            wf.flush()
        mean_val_loss = np.mean(min_loss_val_list)
        test_results_list = np.array(test_results_list)
        test_auc_mean = np.mean(test_results_list[:, 0])
        test_prec_mean = np.mean(test_results_list[:, 1])
        test_rec_mean = np.mean(test_results_list[:, 2])
        test_f1_mean = 2*test_prec_mean*test_rec_mean/(test_prec_mean+test_rec_mean)
        wf.write("from src {}, avg min valid loss {:.4f}, best test metrics: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n".format(
            src, mean_val_loss, test_auc_mean, test_prec_mean, test_rec_mean, test_f1_mean
        ))
        wf.write("\n")
    wf.write(json.dumps(vars(args)) + "\n")
    wf.close()


if __name__ == "__main__":
    train(args)
    pass

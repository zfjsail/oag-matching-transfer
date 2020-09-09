from os.path import join
import sys
import time
import argparse
import numpy as np
import json
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F

from dataset import ProcessedRNNInputDataset
from models.rnn import BiLSTM
from utils.data_utils import ChunkSampler
from utils import settings

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', stream=sys.stdout)  # include timestamp

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnn', help="models used")
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--delta-seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--train-num', default=None, help='Number of training samples')
parser.add_argument('--lr', type=float, default=3e-5, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=1e-3,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--embedding-size', type=int, default=128,
                    help="Embeding size for LSTM layer")
parser.add_argument('--hidden-size', type=int, default=32,
                    help="Hidden size for LSTM layer")
parser.add_argument('--max-sequence-length', type=int, default=17,
                    help="Max sequence length for raw sequences")
parser.add_argument('--max-key-sequence-length', type=int, default=8,
                    help="Max key sequence length for key sequences")
parser.add_argument('--batch', type=int, default=32, help="Batch size")
parser.add_argument('--shuffle', action='store_true', default=True, help="Shuffle dataset")
parser.add_argument('--file-dir', type=str, default=settings.VENUE_DATA_DIR, help="Input file directory")
parser.add_argument('--entity-type', type=str, default="venue", help="entity type to match")

parser.add_argument('--check-point', type=int, default=3, help="Check point")
parser.add_argument('--multiple', type=int, default=16, help="decide how many times to multiply a scalar input")
parser.add_argument('--n-try', type=int, default=1, help="Repeat Times")

args = parser.parse_args()


def evaluate(epoch, loader, model, writer, thr=None, return_best_thr=False, args=args):
    model.eval()
    total = 0.
    loss = 0.
    y_true, y_pred, y_score = [], [], []

    for ibatch, batch in enumerate(loader):
        labels = batch[-1]
        bs = len(labels)

        if args.cuda:
            batch = [data.cuda() for data in batch]
            labels = labels.cuda()
        output, _ = model(batch[0], batch[1], batch[2], batch[3])
        loss_batch = F.nll_loss(output, labels)
        loss += bs * loss_batch.item()
        y_true += labels.data.tolist()
        y_pred += output.max(1)[1].data.tolist()
        y_score += output[:, 1].data.tolist()
        total += len(labels)

    model.train()

    if thr is not None:
        print("using threshold %.4f" % thr)
        y_score = np.array(y_score)
        y_pred = np.zeros_like(y_score)
        y_pred[y_score > thr] = 1

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info("loss: %.4f AUC: %.4f Prec: %.4f Rec: %.4f F1: %.4f",
                loss / total, auc, prec, rec, f1)

    if return_best_thr:  # valid
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        f1s = f1s[:-1]
        thrs = thrs[~np.isnan(f1s)]
        f1s = f1s[~np.isnan(f1s)]
        best_thr = thrs[np.argmax(f1s)]
        logger.info("best threshold=%4f, f1=%.4f", best_thr, np.max(f1s))

        writer.add_scalar('val_loss',
                          loss / total,
                          epoch)

        return best_thr, [loss / total, auc, prec, rec, f1]
    else:
        writer.add_scalar('test_f1',
                          f1,
                          epoch)
        return None, [loss / total, auc, prec, rec, f1]


def train(epoch, train_loader, valid_loader, test_loader, model, optimizer, writer, args=args):
    model.train()

    loss = 0.
    total = 0.

    for i_batch, batch in enumerate(train_loader):
        bs = batch[-1].shape[0]
        labels = batch[-1]

        if args.cuda:
            batch = [data.cuda() for data in batch]
            labels = labels.cuda()
        optimizer.zero_grad()
        output, _ = model(batch[0], batch[1], batch[2], batch[3])
        loss_train = F.nll_loss(output, labels)
        loss += bs * loss_train.item()
        total += bs
        loss_train.backward()
        optimizer.step()
    logger.info("train loss epoch %d: %f", epoch, loss / total)

    writer.add_scalar('training_loss',
                      loss / total,
                      epoch)

    metrics_val = None
    metrics_test = None
    if (epoch + 1) % args.check_point == 0:
        logger.info("epoch %d, checkpoint! validation...", epoch)
        best_thr, metrics_val = evaluate(epoch, valid_loader, model, writer, return_best_thr=True, args=args)
        logger.info('eval on test data!...')
        _, metrics_test = evaluate(epoch, test_loader, model, writer, thr=best_thr, args=args)

    return metrics_val, metrics_test


def train_one_time(args, wf, repeat_seed):
    writer = SummaryWriter('runs/{}_rnn_train_ratio_{}_{}'.format(args.entity_type, args.train_num, repeat_seed))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    logger.info('cuda is available %s', args.cuda)

    dataset = ProcessedRNNInputDataset(args.entity_type, "train", args.train_num, repeat_seed)
    dataset_valid = ProcessedRNNInputDataset(args.entity_type, "valid")
    dataset_test = ProcessedRNNInputDataset(args.entity_type, "test")
    N = len(dataset)
    N_valid = len(dataset_valid)
    N_test = len(dataset_test)
    print("n_train", N)
    train_loader = DataLoader(dataset, batch_size=args.batch,
                              sampler=ChunkSampler(N, 0))
    valid_loader = DataLoader(dataset_valid, batch_size=args.batch,
                              sampler=ChunkSampler(N_valid, 0))
    test_loader = DataLoader(dataset_test, batch_size=args.batch,
                             sampler=ChunkSampler(N_test, 0))

    np.random.seed(args.seed + repeat_seed)
    torch.manual_seed(args.seed + repeat_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + repeat_seed)

    pretrain_emb = torch.load(join(settings.OUT_DIR, "rnn_init_word_emb.emb"))

    model = BiLSTM(vocab_size=settings.MAX_WORD_TOKEN_NUM,
                   pretrain_emb=pretrain_emb,
                   embedding_size=args.embedding_size,
                   hidden_size=args.hidden_size,
                   dropout=args.dropout)
    print(model)
    n_paras = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of paras:", n_paras)

    if args.cuda:
        model.cuda()
    model = model.float()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t_total = time.time()
    logger.info("training...")

    loss_val_min = None
    best_test_metric = None
    model_dir = join(settings.OUT_DIR, args.entity_type)

    for epoch in range(args.epochs):
        print("training epoch", epoch)
        metrics = train(epoch, train_loader, valid_loader, test_loader, model, optimizer, writer, args=args)

        metrics_val, metrics_test = metrics
        if metrics_val is not None:
            if loss_val_min is None or metrics_val[0] < loss_val_min:
                loss_val_min = metrics_val[0]
                best_test_metric = metrics_test
                best_model = model
                torch.save(best_model.state_dict(),
                           join(model_dir, "rnn-match-best-now-train-num-{}.mdl".format(args.train_num)))

    logger.info("optimization Finished!")
    logger.info("total time elapsed: {:.4f}s".format(time.time() - t_total))

    print("min_val_loss", loss_val_min, "best test metrics", best_test_metric[1:])
    # with open(join(model_dir, "{}_rnn_train_ratio_{}_results.txt".format(args.entity_type, args.train_num)), "w") as wf:
    wf.write(
        "min valid loss {:.4f}, best test metrics: AUC: {:.2f}, Prec: {:.2f}, Rec: {:.2f}, F1: {:.2f}\n\n".format(
            loss_val_min, best_test_metric[1] * 100, best_test_metric[2] * 100, best_test_metric[3] * 100,
                          best_test_metric[4] * 100
        ))
    # wf.write(json.dumps(vars(args)) + "\n")
    writer.close()


def main(args):
    model_dir = join(settings.OUT_DIR, args.entity_type)
    wf = open(join(model_dir, "{}_rnn_train_num_{}_results.txt".format(args.entity_type, args.train_num)), "w")
    for t in range(args.n_try):
        train_one_time(args, wf, t)
        wf.flush()
    wf.write(json.dumps(vars(args)) + "\n")
    wf.close()


if __name__ == '__main__':
    main(args=args)

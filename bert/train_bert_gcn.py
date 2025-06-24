import pandas as pd
import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from ignite.metrics import MetricsLambda
from finetune_bert import get_init_local_bert_path, get_predict_lable_dic, save_test_dataset_data

from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
from sklearn.metrics import accuracy_score
import numpy as np
import os
import shutil
import argparse
import sys
import logging
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score

from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.metrics import Metric
from sklearn.metrics import cohen_kappa_score

# Training
# Define F1 score using precision and recall
def F1(precision, recall):
    return (2 * precision * recall / (precision + recall + 1e-20))

def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


def training_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def evaluation_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true

# define cohenkappa
class CohenKappa(Metric):
    def __init__(self, output_transform=lambda x: x):
        super(CohenKappa, self).__init__(output_transform=output_transform)
        self._predictions = []
        self._targets = []

    def reset(self):
        self._predictions = []
        self._targets = []
        super(CohenKappa, self).reset()

    def update(self, output):
        y_pred, y = output
        y_pred = th.argmax(y_pred, dim=1)
        self._predictions.extend(y_pred.cpu().numpy())
        self._targets.extend(y.cpu().numpy())

    def compute(self):
        global y_test_pred_results, y_test_true_results
        y_test_pred_results = self._predictions
        y_test_true_results = self._targets
        return cohen_kappa_score(self._targets, self._predictions)





if __name__ == '__main__':
    global y_test_pred_results, y_test_true_results,is_save_model
    y_test_pred_results = []
    y_test_true_results = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--balance',  type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
    parser.add_argument('--nb_epochs', type=int, default=50)
    parser.add_argument('--bert_init', type=str, default='roberta-base')
    parser.add_argument('--pretrained_bert_ckpt', default=None)
    parser.add_argument('--dataset', default='20ng', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'we','we_test'])
    parser.add_argument('--checkpoint_dir', default=None,
                        help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
    parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--n_hidden', type=int, default=200,
                        help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
    parser.add_argument('--heads', type=int, default=8, help='the number of attention n heads for gat')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--gcn_lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--is_save_model', type=str, default='save',choices=['save', 'no_save'], help='input save or no_save')

    args = parser.parse_args()
    max_length = args.max_length
    batch_size = args.batch_size
    balance = args.balance
    nb_epochs = args.nb_epochs
    bert_init = args.bert_init
    pretrained_bert_ckpt = args.pretrained_bert_ckpt
    dataset = args.dataset
    checkpoint_dir = args.checkpoint_dir
    gcn_model = args.gcn_model
    gcn_layers = args.gcn_layers
    n_hidden = args.n_hidden
    heads = args.heads
    dropout = args.dropout
    gcn_lr = args.gcn_lr
    bert_lr = args.bert_lr
    is_save_model = args.is_save_model

    if checkpoint_dir is None:
        ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
    else:
        ckpt_dir = checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    shutil.copy(os.path.basename(__file__), ckpt_dir)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter('%(message)s'))
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
    fh.setFormatter(logging.Formatter('%(message)s'))
    fh.setLevel(logging.INFO)
    logger = logging.getLogger('training logger')
    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    logger.info('arguments:')
    logger.info(str(args))
    logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
    if is_save_model == 'no_save':
        logger.info(f'The user chooses to save only the training results without saving the model checkpoints!')

    cpu = th.device('cpu')
    gpu = th.device('cuda:0')
    # Model

    # Data Preprocess
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
    '''
    adj: n*n sparse adjacency matrix
    y_train, y_val, y_test: n*c matrices 
    train_mask, val_mask, test_mask: n-d bool array
    '''

    # compute number of real train/val/test/word nodes and number of classes
    nb_node = features.shape[0]
    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_word = nb_node - nb_train - nb_val - nb_test
    nb_class = y_train.shape[1]
    pretrained_model_path = get_init_local_bert_path(bert_init)

    # instantiate model according to class number
    if gcn_model == 'gcn':
        model = BertGCN(nb_class=nb_class, pretrained_model=pretrained_model_path, m=balance, gcn_layers=gcn_layers,
                        n_hidden=n_hidden, dropout=dropout)
    else:
        model = BertGAT(nb_class=nb_class, pretrained_model=pretrained_model_path, m=balance, gcn_layers=gcn_layers,
                        heads=heads, n_hidden=n_hidden, dropout=dropout)

    if pretrained_bert_ckpt is not None:
        ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
        model.bert_model.load_state_dict(ckpt['bert_model'])
        model.classifier.load_state_dict(ckpt['classifier'])

    # load documents and compute input encodings
    corpse_file = './data/corpus/' + dataset + '_shuffle.txt'
    with open(corpse_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')


    def encode_input(text, tokenizer):
        input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
        #     print(input.keys())
        return input.input_ids, input.attention_mask


    input_ids, attention_mask = encode_input(text, model.tokenizer)
    input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
    attention_mask = th.cat(
        [attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

    # transform one-hot label to class ID for pytorch computation
    y = y_train + y_test + y_val
    y_train = y_train.argmax(axis=1)
    y = y.argmax(axis=1)

    # document mask used for update feature
    doc_mask = train_mask + val_mask + test_mask

    # build DGL Graph
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
    g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
    g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
        th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
    g.ndata['label_train'] = th.LongTensor(y_train)
    g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

    logger.info('graph information:')
    logger.info(str(g))

    # create index loader
    train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
    val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
    test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))

    doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

    idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
    idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
    idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
    idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

    optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)  # decay lr 20% after every step

    trainer = Engine(training_step)

    evaluator = Engine(evaluation_step)
    metrics={
        'acc': Accuracy(),
        'nll': Loss(th.nn.CrossEntropyLoss()),
        'precision': Precision(average='macro'),
        'recall': Recall(average='macro'),
        'kappa': CohenKappa()
    }
    f1_score = MetricsLambda(F1, metrics['precision'], metrics['recall'])
    f1_score.attach(evaluator, 'f1_score')

    for n, f in metrics.items():
        f.attach(evaluator, n)
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        global y_test_pred_results, y_test_true_results, is_save_model
        evaluator.run(idx_loader_train)
        metrics = evaluator.state.metrics
        train_acc, train_nll, train_precision, train_recall, train_f1_score, train_kappa = metrics["acc"], metrics["nll"], metrics[
            "precision"], metrics["recall"], metrics["f1_score"], metrics["kappa"]
        evaluator.run(idx_loader_val)
        metrics = evaluator.state.metrics
        val_acc, val_nll, val_precision, val_recall, val_f1_score, val_kappa = metrics["acc"], metrics["nll"], metrics[
            "precision"], metrics["recall"], metrics["f1_score"], metrics["kappa"]
        evaluator.run(idx_loader_test)
        metrics = evaluator.state.metrics
        test_acc, test_nll, test_precision, test_recall, test_f1_score, test_kappa = metrics["acc"], metrics["nll"], \
                                                                                     metrics["precision"], metrics[
                                                                                         "recall"], metrics["f1_score"], \
                                                                                     metrics["kappa"]

        logger.info(
            "\rEpoch: {} Train dataset: Train acc: {:.4f} train_loss: {:.4f}  train_precision: {:.4f} train_recall: {:.4f}  train_f1_score: {:.4f} train_kappa: {:.4f}"
                .format(trainer.state.epoch, train_acc, train_nll, train_precision, train_recall, train_f1_score, train_kappa)
        )
        logger.info(
            "\rEpoch: {} Valid dataset: Valid_acc: {:.4f} Valid_loss: {:.4f}  Val_precision: {:.4f} val_recall: {:.4f}  val_f1_score: {:.4f} val_kappa: {:.4f}"
                .format(trainer.state.epoch, val_acc, val_nll, val_precision, val_recall, val_f1_score, val_kappa)
        )
        logger.info(
            "\rEpoch: {} Test dataset: Test_acc: {:.4f} Test_loss: {:.4f}  Test_precision: {:.4f} Test_recall: {:.4f}  Test_f1_score : {:.4f} Test_kappa: {:.4f}"
                .format(trainer.state.epoch, test_acc, test_nll, test_precision, test_recall, test_f1_score, test_kappa)
        )
        # Read label_list
        lable_list_path = "./data/corpus/we_labels.txt".replace("we", dataset)
        labels_dic = get_predict_lable_dic(lable_list_path)
        # Replace the values in y_preds and y_trues with labels_dic
        y_test_pred_results = [labels_dic[pred] for pred in y_test_pred_results]
        y_test_true_results = [labels_dic[true] for true in y_test_true_results]

        prediction_data = pd.DataFrame({'label_': y_test_true_results, 'label_preds': y_test_pred_results})

        if log_training_results.best_val_kappa == 0 or log_training_results.best_val_kappa < val_kappa:
            log_training_results.best_val_kappa = val_kappa
            predicted_test_data_path = os.path.join(
                ckpt_dir, 'predicted_test_data.xlsx'
            )
            save_test_dataset_data(prediction_data,predicted_test_data_path,dataset)
            logger.info("New checkpoint")
            if is_save_model=="save":
                th.save(
                    {
                        'bert_model': model.bert_model.state_dict(),
                        'classifier': model.classifier.state_dict(),
                        'gcn': model.gcn.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': trainer.state.epoch,
                    },
                    os.path.join(
                        ckpt_dir, 'checkpoint.pth'
                    )
                )
            scheduler.step()

        else:
            logger.info('---Old performance {} > New performance {}---'.format(round(log_training_results.best_val_kappa, 4), round(val_kappa, 4)))
            logger.info('No improvement detected compared to last validation round, early stop is triggered.')
            exit()

    log_training_results.best_val_kappa = 0
    g = update_feature()
    trainer.run(idx_loader, max_epochs=nb_epochs)

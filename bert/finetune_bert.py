import pandas as pd
import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
import argparse, shutil, logging
from torch.optim import lr_scheduler
from model.models import BertClassifier, DebertaClassifier
from torch.utils.data import DataLoader, Dataset, Sampler

from sklearn.metrics import roc_curve, roc_auc_score, auc,precision_score, f1_score, accuracy_score, recall_score

from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.engine import Events
from ignite.metrics import MetricsLambda
from ignite.metrics import Metric
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Subset, RandomSampler
from data.coauthor.coathor_to_train_data import reorder_dataframe
import warnings


def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding=True, return_tensors='pt')
    return input.input_ids, input.attention_mask

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

def training_step(engine, batch):
    global model, optimizer, train_data
    model.train()
    model = model.to(gpu)
    optimizer.zero_grad()
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc

def evaluation_step(engine, batch):
    global model
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label

        return y_pred, y_true

# Define F1 score using precision and recall
def F1(precision, recall):
    return (2 * precision * recall / (precision + recall + 1e-20))

def get_predict_lable_dic(lable_list_path):
    labels_dic = {}
    with open(lable_list_path, 'r') as file:
        for idx, line in enumerate(file):
            labels_dic[idx] = int(line.strip())
    return labels_dic


def read_dataset_xlsx(data_path):
    data_table = pd.read_excel(data_path, engine='openpyxl')

    selected_data_table = data_table[
        (data_table['label'].isin([0, 1, 2])) &
        (data_table['train_ix'].isin(['train', 'test', 'valid']))
        ].copy()
    selected_data_table_sorted = reorder_dataframe(selected_data_table, "train_ix")
    test_dataset = selected_data_table_sorted.loc[
        selected_data_table_sorted['train_ix'] == 'test'
        ].copy()
    test_dataset = test_dataset.reset_index(drop=True)
    return test_dataset

# Customize the Sampler

class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        # Randomly select num_samples of different indexes
        return iter(th.randperm(len(self.data_source)).tolist()[:self.num_samples])
    def __len__(self):
        return self.num_samples

def save_test_dataset_data(prediction_data,merged_data_path,dataset):
    if dataset == "we":
        data_table_path = 'data/coauthor/20231114_coauthor_data.xlsx'
    elif dataset == "we_test":
        data_table_path = 'data/coauthor/20231114_coauthor_data_test.xlsx'
    data_table = read_dataset_xlsx(data_table_path)
    merged_data = pd.concat([data_table.reset_index(drop=True), prediction_data.reset_index(drop=True)], axis=1)
    # Save as xlsx file
    merged_data.to_excel(merged_data_path, index=False)

def get_init_local_bert_path(bert_init):
    # Create a dictionary that maps the model name to the path of the pre-trained model
    pretrained_model_paths = {
        "roberta-base": "FacebookAI/roberta-base",
        "bert-base-uncased": "//liusanya/liushiqi/projects/ACSA_HGCN/bert_models/bert-base-uncased",
        "distilbert-base-uncased": "./pre_trained_model/distilbert-base-uncased",
        "deberta-v3-base":"./pre_trained_model/deberta-v3-base",
        "deberta-base":"./pre_trained_model/deberta-base",
    }
    if bert_init in pretrained_model_paths:
        # If found, return the corresponding value
        return pretrained_model_paths[bert_init]
    else:
        # If not found, throw an exception and exit the program
        raise ValueError(f"Error: '{bert_init}' not found in the pretrained_model_paths.")
        sys.exit(1)

if __name__ == '__main__':
    # Used to save prediction results
    global y_test_pred_results, y_test_true_results, is_save_model
    y_test_pred_results = []
    y_test_true_results = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=60)
    parser.add_argument('--bert_lr', type=float, default=1e-4)
    parser.add_argument('--is_save_model', type=str, default='save',choices=['save', 'no_save'], help='input save or no_save')

    # we = write essay
    parser.add_argument('--dataset', default='we_test', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', "we","we_test"])
    parser.add_argument('--epoch_sample_num', type=int, default=20000)

    parser.add_argument('--bert_init', type=str, default='roberta-base',choices=["roberta-base", "bert-base-uncased", "microsoft/deberta-v3-base","deberta-v3-base","deberta-base"])
    parser.add_argument('--checkpoint_dir', default=None,
                        help='checkpoint directory, [bert_init]_[dataset] if not specified')
    args = parser.parse_args()

    max_length = args.max_length
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    bert_lr = args.bert_lr
    dataset = args.dataset
    bert_init = args.bert_init
    checkpoint_dir = args.checkpoint_dir
    epoch_sample_num = args.epoch_sample_num
    is_save_model = args.is_save_model

    if checkpoint_dir is None:
        ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
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

    cpu = th.device('cpu')
    gpu = th.device('cuda:0')

    logger.info('arguments:')
    logger.info(str(args))
    logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
    if is_save_model == 'no_save':
        logger.info(f'The user chooses to save only the training results without saving the model checkpoints!')

    # Data Preprocess
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
    '''
    y_train, y_val, y_test: n*c matrices 
    train_mask, val_mask, test_mask: n-d bool array
    train_size, test_size: unused
    '''

    # compute number of real train/val/test/word nodes and number of classes
    nb_node = adj.shape[0]
    nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    nb_word = nb_node - nb_train - nb_val - nb_test
    nb_class = y_train.shape[1]
    # use local model
    # pretrained_model_path = get_init_local_bert_path(bert_init)
    pretrained_model_path = bert_init
    # instantiate model according to class number
    # model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)
    logger.info('reading model from path: {}'.format(pretrained_model_path))
    if "deberta" in bert_init:
        model=DebertaClassifier(pretrained_model=pretrained_model_path, nb_class=nb_class)
        warnings.filterwarnings("ignore", category=UserWarning)

    else:
        model = BertClassifier(pretrained_model=pretrained_model_path, nb_class=nb_class)

    # transform one-hot label to class ID for pytorch computation
    y = th.LongTensor((y_train + y_val + y_test).argmax(axis=1))
    label = {}
    label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train + nb_val], y[-nb_test:]

    # load documents and compute input encodings
    corpus_file = './data/corpus/' + dataset + '_shuffle.txt'
    with open(corpus_file, 'r') as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')

    input_ids, attention_mask = {}, {}

    input_ids_, attention_mask_ = encode_input(text, model.tokenizer)

    # create train/test/val datasets and dataloaders
    input_ids['train'], input_ids['val'], input_ids['test'] = input_ids_[:nb_train], input_ids_[
                                                                                     nb_train:nb_train + nb_val], input_ids_[
                                                                                                                  -nb_test:]
    attention_mask['train'], attention_mask['val'], attention_mask['test'] = attention_mask_[
                                                                             :nb_train], attention_mask_[
                                                                                         nb_train:nb_train + nb_val], attention_mask_[
                                                                                                                      -nb_test:]
    datasets = {}
    loader = {}

    for split in ['train', 'val', 'test']:
        datasets[split] = Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
        if split in ['test', 'val']:
            loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=False)
        else:
            sampler = RandomSubsetSampler(datasets[split], num_samples=epoch_sample_num)
            print(f"train dataset sample num: {epoch_sample_num}")
            loader[split] = DataLoader(datasets[split], batch_size=batch_size, sampler=sampler)
            # print("sample no")

    # define train and test function

    optimizer = th.optim.Adam(model.parameters(), lr=bert_lr)
    # decay lr 20% after every step
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.8)
    trainer = Engine(training_step)

    evaluator = Engine(evaluation_step)
    metrics={
        'acc': Accuracy(),
        'nll': Loss(th.nn.CrossEntropyLoss()),
        'precision': Precision(average='macro'),
        'recall': Recall(average='macro'),
        'kappa': CohenKappa()  # Use the custom cohen kappa function
    }
    f1_score = MetricsLambda(F1, metrics['precision'], metrics['recall'])
    f1_score.attach(evaluator, 'f1_score')

    for n, f in metrics.items():
        f.attach(evaluator, n)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        global y_test_pred_results, y_test_true_results, is_save_model
        evaluator.run(loader['train'])
        metrics = evaluator.state.metrics
        train_acc, train_nll, train_precision, train_recall, train_f1_score, train_kappa = metrics["acc"], metrics["nll"], metrics[
            "precision"], metrics["recall"], metrics["f1_score"], metrics["kappa"]
        evaluator.run(loader['val'])
        metrics = evaluator.state.metrics
        val_acc, val_nll, val_precision, val_recall, val_f1_score, val_kappa = metrics["acc"], metrics["nll"], metrics[
            "precision"], metrics["recall"], metrics["f1_score"], metrics["kappa"]
        evaluator.run(loader['test'])
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
        # Use labels_dic to replace values in y_preds and y_trues
        y_test_pred_results = [labels_dic[pred] for pred in y_test_pred_results]
        y_test_true_results = [labels_dic[true] for true in y_test_true_results]
        # Create a DataFrame
        prediction_data = pd.DataFrame({'label_': y_test_true_results,'label_preds': y_test_pred_results})

        if log_training_results.best_val_kappa == 0 or log_training_results.best_val_kappa < val_kappa:
            log_training_results.best_val_kappa = val_kappa
            logger.info("New checkpoint")
            predicted_test_data_path = os.path.join(
                ckpt_dir, 'predicted_test_data.xlsx'
            )
            save_test_dataset_data(prediction_data,predicted_test_data_path,dataset)
            # 保存模型
            if is_save_model=="save":
                th.save(
                    {
                        'bert_model': model.bert_model.state_dict(),
                        'classifier': model.classifier.state_dict(),
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
    # For each epoch, RandomSampler will randomly select n samples from the dataset, because we defined the sampler earlier.
    trainer.run(loader['train'], max_epochs=nb_epochs)
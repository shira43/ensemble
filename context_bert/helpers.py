import argparse
import logging
from typing import Optional, Literal
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss, Accuracy
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AdamW
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
from torch.nn.functional import cross_entropy
from ignite.metrics import Precision, Recall, Fbeta, Metric
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate, numpy as np
from transformers import BertForSequenceClassification
from ignite.engine import Engine, Events

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

cpu = torch.device('cpu')
gpu = torch.device('cuda:1')

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
        y_pred = torch.argmax(y_pred, dim=1)
        self._predictions.extend(y_pred.cpu().numpy())
        self._targets.extend(y.cpu().numpy())

    def compute(self):
        global y_test_pred_results, y_test_true_results
        y_test_pred_results = self._predictions
        y_test_true_results = self._targets
        return cohen_kappa_score(self._targets, self._predictions)


# def get_collate_fn(tokenizer):
#     def collate_fn(batch):
#         return tokenizer.pad(batch, return_tensors="pt")
#     return collate_fn



def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=128,
                weighted_sampler=True, epoch_sample_num: Optional[int] = None, collate_fn=None):
    """
        Build   train_loader  (with or without WeightedRandomSampler)
           val_loader     (no shuffle)
           test_loader    (no shuffle)
           train_eval_loader (same order as val/test, handy for eval-on-train)

    :param train_dataset: all dataset are Pytorch Datasets with input_ids, token_type_ids, attention_mask and labels keys
    :param val_dataset: ...
    :param test_dataset:  ...
    :param batch_size: batch_size for dataLoader
    :param weighted_sampler: True if weighted sampler should be applied (for unbalanced datasets)
    :param epoch_sample_num: through how many train samples it iterates in one epoch
                                -> default = None -> (every sample once)
    :param collate_fn: function to tell Dataloader how to merge a list of samples into a single batch.

    :return: Returns dict of Dataloaders keyed by split name.
    """


    if epoch_sample_num is None:
        epoch_sample_num = len(train_dataset)

    if weighted_sampler:
        logger.info("Computing Class weights from the train set now...")
        # compute class weights from the train dataset
        train_labels = np.array(train_dataset["labels"])
        nb_class = int(max(train_labels)) + 1
        counts = np.bincount(train_labels, minlength=nb_class)
        class_weights = 1.0 / counts
        sample_weights = torch.as_tensor(class_weights[train_labels], dtype=torch.float)

        logger.info("Successfully computed class weights. \n Initializing Weighted Sampler...")


        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=epoch_sample_num,
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    logger.info("Now building and Saving Dataloaders.")
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,   # no shuffle when sampler is set
            sampler=sampler,
            drop_last=True,  # ensures every batch is full-size
            collate_fn=collate_fn,
        ),

        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),

        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),

        # Same data as 'train' but deterministic order (useful for evaluation)
        "train_eval": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
    }

    return loaders


def training_step(engine, batch, model, optimizer):

    model.train()
    batch = {k: v.to(gpu) for k, v in batch.items()}  # move batch to gpu

    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        labels=batch["labels"]  # one integer per example
    )

    loss = outputs.loss  # already CrossEntropyLoss
    logits = outputs.logits.detach().cpu()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return {
        "loss": loss.item(),
        "y_pred": logits,
        "y_true": batch["labels"].detach().cpu()
    }


def evaluation_step(engine, batch, model):
    model.eval()
    with torch.no_grad():
        batch = {k: v.to(gpu) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"]
        )

        return outputs.logits, batch["labels"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--use_weighted_sampler', type=bool, default=False)
    parser.add_argument('--nb_epochs', type=int, default=7)
    parser.add_argument('--bert_lr', type=float, default=1e-4)
    parser.add_argument('--is_save_model', type=str, default='save', choices=['save', 'no_save'],
                        help='input save or no_save')
    parser.add_argument('--dataset', default='coauthor-zeng',
                        choices=["pasted-base", "coauthor-base", "coauthor-extended-base", "coauthor-zeng"])
    parser.add_argument('--epoch_sample_num', type=int, default=None)


    parser.add_argument('--bert_init', type=str, default='bert-base-uncased', choices=["roberta-base", "bert-base-uncased",
                                                                                  "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                                                                  "microsoft/deberta-v3-base",
                                                                                  "deberta-v3-base", "deberta-base"])
    parser.add_argument('--checkpoint_dir', default=None,
                        help='checkpoint directory, [bert_init]_[dataset] if not specified')

    # only for finetune_context_bert
    parser.add_argument('--max_context_sentences', type=int, default=2)

    #only for finetune_sequence_bert
    parser.add_argument('--max_length', type=int, default=512, help='the max input length for bert')

    return parser.parse_args()


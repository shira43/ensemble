import argparse
from typing import Optional

from ignite.metrics import Loss, Accuracy
from jieba.lac_small.predict import batch_size
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer, AdamW
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
from torch.nn.functional import cross_entropy
from ignite.metrics import Precision, Recall, Fbeta, Metric
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate, numpy as np
from transformers import BertForSequenceClassification
from ignite.engine import Engine, Events
from BertContextDataset import BertContextDataset


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


def collate_fn(batch):
    return tokenizer.pad(
        batch,
        return_tensors="pt",
    )


def tokenization(example):
    text_col = 'sentence_text' if 'sentence_text' in example else 'text'
    return tokenizer(example[text_col], add_special_tokens=False)


def training_step(engine, batch):
    global model, optimizer, train_data
    model.train()
    optimizer.zero_grad()

    # Forward pass – BertForSequenceClassification returns (loss, logits)
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        token_type_ids=batch["token_type_ids"],
        labels=batch["label"]  # one integer per example
    )

    loss = outputs.loss  # already CrossEntropyLoss
    logits = outputs.logits

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return {
        "loss": loss.item(),
        "y_pred": logits.detach(),
        "y_true": batch["label"].detach()
    }


def evaluation_step(engine, batch):
    global model
    model.eval()
    with torch.no_grad():
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"]
        )

        return outputs.logits, batch["label"]



def filter_and_tokenize(data):
    """

    :param data: huggingface dataset with splits "train", "validation", "test", and either a column "sentence_text" or "text"
    :return: three instances of BertContextDataset each for train, validation, test
    """
    # get split and keep only rows with label == 0, 1 or 2
    train_set = data["train"].filter(lambda example: example["label"] in [0, 1, 2])
    val_set = data["validation"].filter(lambda example: example["label"] in [0, 1, 2])
    test_set = data["test"].filter(lambda example: example["label"] in [0, 1, 2])

    train_set = train_set.map(tokenization, batched=True)
    val_set = val_set.map(tokenization, batched=True)
    test_set = test_set.map(tokenization, batched=True)

    train_dataset = BertContextDataset(train_set, tokenizer)
    val_dataset = BertContextDataset(val_set, tokenizer)
    test_dataset = BertContextDataset(test_set, tokenizer)

    print("Filtering and tokenization of dataset completed.")

    return train_dataset, val_dataset, test_dataset


def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=batch_size,
                weighted_sampler=True, epoch_sample_num: Optional[int] = None):
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

    :return: Returns dict of Dataloaders keyed by split name.
    """

    # compute class weights from the train dataset
    train_labels = [example["label"] for example in train_dataset]
    nb_class = int(max(train_labels)) + 1
    counts = np.bincount(train_labels, minlength=nb_class)
    class_weights = 1.0 / counts
    sample_weights = torch.as_tensor(class_weights[train_labels], dtype=torch.float)

    if epoch_sample_num is None:
        epoch_sample_num = len(train_dataset)

    if weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=epoch_sample_num,
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

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


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--max_length', type=int, default=512, help='the input length for bert')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--use_weighted_sampler', type=bool, default=True)
    parser.add_argument('--nb_epochs', type=int, default=7)
    parser.add_argument('--bert_lr', type=float, default=1e-4)
    parser.add_argument('--is_save_model', type=str, default='save', choices=['save', 'no_save'],
                        help='input save or no_save')
    parser.add_argument('--dataset', default='coauthor-zeng',
                        choices=["pasted-base", "coauthor-base", "coauthor-extended-base", "coauthor-zeng"])
    parser.add_argument('--epoch_sample_num', type=int, default=None)

    parser.add_argument('--bert_init', type=str, default='roberta-base', choices=["roberta-base", "bert-base-uncased",
                                                                                  "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                                                                                  "microsoft/deberta-v3-base",
                                                                                  "deberta-v3-base", "deberta-base"])
    parser.add_argument('--checkpoint_dir', default=None,
                        help='checkpoint directory, [bert_init]_[dataset] if not specified')

    return parser.parse_args()


if __name__ == "__main__":
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:0')

    args = parse_args()

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    bert_lr = args.bert_lr
    use_weighted_sampler = args.use_weighted_sampler
    # is_save_model = args.is_save_model
    dataset = args.dataset
    # bert_init = args.bert_init
    # checkpoint_dir = args.checkpoint_dir
    epoch_sample_num = args.epoch_sample_num

    # if checkpoint_dir is None:
    #     ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
    # else:
    #     ckpt_dir = checkpoint_dir

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = load_dataset(f"43shira43/{dataset}",cache_dir="/tmp/hf_cache")

    train, val, test = filter_and_tokenize(dataset)

    dataloaders = get_loaders(train, val, test, batch_size, use_weighted_sampler, epoch_sample_num)

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    train_eval_loader = dataloaders["train_eval"]

    print("finished. Loading Dataloader now.")


    # TODO testen ob evtl für token classification bessere ergebnisse erzielt werden ?
    #
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        problem_type="single_label_classification"
    )
    model = model.to(gpu)

    optimizer = AdamW(model.parameters(), lr=bert_lr)

    trainer = Engine(training_step)
    evaluator = Engine(evaluation_step)


    # attach metrics
    # trainer: batch-level loss/acc aggregated per epoch
    Loss(lambda o: o["loss"]).attach(trainer, "loss")
    Accuracy(output_transform=lambda o: (o["y_pred"], o["y_true"])).attach(trainer, "acc")

    # evaluator: full-epoch metrics on val_loader and train_loader
    metrics = {
        "loss": Loss(cross_entropy),
        "acc": Accuracy(),
        "precision": Precision(average=True),  # micro-avg over classes
        "recall": Recall(average=True),
        "f1": Fbeta(beta=1.0, average=True),
        "kappa": CohenKappa()
    }

    for name, metric in metrics.items():
        metric.attach(evaluator, name)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_train_metrics(engine):
        m = engine.state.metrics
        print(f"[Train] Epoch {engine.state.epoch:02d} | "
              f"loss={m['loss']:.4f} | acc={m['acc']:.4f}")


    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        evaluator.run(val_loader)
        m = evaluator.state.metrics
        print(f"[Val]   Epoch {engine.state.epoch:02d} | "
              f"loss={m['loss']:.4f} | "
              f"acc={m['acc']:.4f} | "
              f"P={m['precision']:.4f} R={m['recall']:.4f} "
              f"F1={m['f1']:.4f} κ={m['kappa']:.4f}")


    trainer.run(train_loader, max_epochs=nb_epochs)

    # ─── after training finishes ───────────────────────────────────────────────────
    model.eval()
    model.bert.config.output_attentions = True  # ask BERT to return attentions

    sample = next(iter(test_loader))  # one random batch
    sample = {k: v.to(gpu) for k, v in sample.items() if k in ("input_ids", "attention_mask", "token_type_ids")}

    with torch.no_grad():
        # run *raw* BERT to get attention maps
        outputs = model.bert(**sample, output_attentions=True, return_dict=True)
        last_layer_attn = outputs.attentions[-1]  # (B, heads, L, L)

    # Inspect or visualise e.g. CLS-to-tokens attention of head-0
    # For a quick print you might do:
    cls_attn = last_layer_attn[0, 0, 0]  # tokens that CLS attends to
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"][0])
    for tok, score, ttype in zip(tokens, cls_attn.cpu().tolist(), sample["token_type_ids"][0].tolist()):
        print(f"{tok:>10}  seg={ttype}  attn={score:.3f}")



    # print("Starting Training loop")
    # trainer = Engine(training_step)
    #
    # # Average loss across epoch
    # Loss(lambda o: o["loss"]).attach(trainer, "loss")
    #
    # # Accuracy across epoch
    # Accuracy(output_transform=lambda o: (o["y_pred"], o["y_true"])).attach(trainer, "acc")
    #
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_epoch(engine):
    #     print(f"Epoch {engine.state.epoch} | "
    #           f"loss={engine.state.metrics['loss']:.4f} | "
    #           f"acc={engine.state.metrics['acc']:.4f}")
    #
    #
    # trainer.run(train_loader, max_epochs=5)
    #
    # evaluator = Engine(evaluation_step)
    # Accuracy().attach(evaluator, "acc")
    # Loss(cross_entropy).attach(evaluator, "loss")
    #
    #
    # # run evaluator every epoch
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def run_validation(engine):
    #     evaluator.run(val_loader)
    #     metrics = evaluator.state.metrics
    #     print(
    #         f"[Val] Epoch {engine.state.epoch} | "
    #         f"loss={metrics['loss']:.4f} | acc={metrics['acc']:.4f}"
    #     )
    #
    #




    # =========================

    # print(train.column_names)
    # print("train input_ids")
    # print(train["input_ids"][:10][0])
    # print(len(train["input_ids"][:10][0]))
    # # # ===================================
    # # # Count how many documents have over 512 tokens.
    # #
    # # currentDoc = ""
    # # docCount = 0
    # # max_len = 0
    # # counter = 0
    # # for element in train:
    # #     doc = element["session_id"]
    # #     input_ids = element["input_ids"]
    # #     if doc != currentDoc:
    # #         if max_len > 512:
    # #             counter += 1
    # #         currentDoc = doc
    # #         docCount += 1
    # #         max_len = 0
    # #         max_len += len(input_ids)
    # #     else:
    # #         max_len += len(input_ids)
    # #
    # #print("Last Document processed: ", currentDoc, "Number of unique session_ids:", docCount, "Length of last doc: ",max_len, "Number of documents over 512 tokens: ", counter)
    #
    #
    # # bert mit satz 1 und 2 tokenizer. Vorbild
    # a, b = get_max_context_sample(
    #     train["input_ids"],
    #     sentence_idx=3,
    #     max_length=tokenizer.max_len_sentences_pair,
    #     max_context_sentences=2,
    #     context_type="prefix",
    # )
    # dict = tokenizer.prepare_for_model(a, b, return_tensors="pt", add_special_tokens=True)
    # print(a)
    # print(b)
    # print(dict)
    # print(sum(i == 0 for i in dict["token_type_ids"]))
    #
    # # CLS ist 101 und SEP 102 (input_ids)
    # train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    # print(train.format['type'])
    #

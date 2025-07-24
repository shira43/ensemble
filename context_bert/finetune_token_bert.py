import logging
from functools import partial

from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss, Accuracy
from transformers import AutoTokenizer, AdamW, BertForTokenClassification, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
import torch
from torch.nn.functional import cross_entropy
from ignite.metrics import Precision, Recall, Fbeta
from ignite.engine import Engine, Events
from helpers import get_loaders, parse_args, CohenKappa, training_step, evaluation_step

# Mix -> "user_and_api" und "O" ist Outside -> "user".
id2label = {
    0: "O",
    1: "B-API",
    2: "I-API",
    3: "B-MIX",
    4: "I-MIX"
}


def tokenization(example):
    text_col = 'sentence_text' if 'sentence_text' in example else 'text'
    return {"tokens": tokenizer(example[text_col],
                                add_special_tokens=False)["input_ids"]}


def build_sequence(ds, max_seq_length=512):
    """

    :param ds: has to have columns "session_id" or "id" and "label"
    :param max_seq_length:
    :return: datasets.Dataset with input_ids, token_type_ids and attention_mask, labels
    """
    # build dataset mit max 512 tokens
    # ds hat column "session_id" oder "id", und tokens column mit input_ids
    id_col = 'session_id' if 'session_id' in ds.column_names else 'id'
    current_doc = None
    sequences = []
    tokens = []
    labels = []

    def flush_sequence():
        if tokens:
            features = tokenizer.prepare_for_model(tokens, add_special_tokens=True, truncation=True, max_length=max_seq_length)
            special_token_mask = tokenizer.get_special_tokens_mask(features["input_ids"],
                                                                   already_has_special_tokens=True)

            # Convert mask to labels: keep labels where token is not special otherwise -100 label
            labels_with_mask = []
            label_idx = 0
            for is_special in special_token_mask:
                if is_special:
                    #TODO change back to -100 maybe. for now its just outside label
                    labels_with_mask.append(0)
                else:
                    labels_with_mask.append(labels[label_idx])
                    label_idx += 1

            features["labels"] = labels_with_mask
            sequences.append(features)

    for row in ds:
        row_id = str(row[id_col])
        if current_doc != row_id:
            flush_sequence()  # save current sequence if docs change
            current_doc = row_id
            tokens = []
            labels = []

        new_tokens = row["tokens"]
        new_labels = []

        if row["label"] == 0:  # Outside
            new_labels = [0] * len(new_tokens)
        elif row["label"] == 1:  # AI
            new_labels = [1] + [2] * (len(new_tokens) - 1)
        elif row["label"] == 2:  # Mixed
            new_labels = [3] + [4] * (len(new_tokens) - 1)

        # If adding would overflow, flush current sequence
        if len(tokens) + len(new_tokens) > max_seq_length:
            flush_sequence()
            tokens = []
            labels = []

        tokens += new_tokens
        labels += new_labels

    flush_sequence()

    return Dataset.from_list(sequences)


def filter_and_tokenize(data, max_seq_length=512):
    """
        :param data: huggingface dataset with splits "train", "validation", "test", and columns "label" and "sentence_text" or "text"
        :param max_seq_length: Maximum number of tokens in a single sequence
        :return: three instances datasets.Dataset with pytorch tensors as values to keys
        """
    # get split and keep only rows with label == 0, 1 or 2
    train_set = data["train"].filter(lambda example: example["label"] in [0, 1, 2])
    val_set = data["validation"].filter(lambda example: example["label"] in [0, 1, 2])
    test_set = data["test"].filter(lambda example: example["label"] in [0, 1, 2])

    # add column tokens which stores input_ids of tokenizer to dataset
    train_set = train_set.map(tokenization, batched=True)
    val_set = val_set.map(tokenization, batched=True)
    test_set = test_set.map(tokenization, batched=True)

    # build sequences per document (with max_seq_length) for the whole dataset
    train_dataset = build_sequence(train_set, max_seq_length )
    val_dataset = build_sequence(val_set, max_seq_length)
    test_dataset = build_sequence(test_set, max_seq_length)

    # turn to torch tensors for dataloader
    cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    for ds in (train_dataset, val_dataset, test_dataset):
        ds.set_format(type="torch", columns=cols)

    return train_dataset, val_dataset, test_dataset


def cls_metric_transform(output):
    """
    output = (logits, labels)  with
        logits : (B, C, S)  after _prepare_logits / permute
        labels : (B, S)
    returns   = (pred_flat, gold_flat)  with only valid positions kept
    """
    y_pred, y = output
    y_pred = torch.argmax(y_pred, dim=1)       # (B, S)

    # keep tokens whose gold label is within range
    mask = (y >= 0) & (y < len(id2label))
    y_pred = y_pred[mask]
    y      = y[mask]
    return y_pred, y


if __name__ == "__main__":
    cpu = torch.device('cpu')
    gpu = torch.device('cuda:1')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    args = parse_args()

    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    bert_lr = args.bert_lr
    use_weighted_sampler = args.use_weighted_sampler
    # is_save_model = args.is_save_model
    dataset = args.dataset
    bert_init = args.bert_init
    max_length = args.max_length
    # checkpoint_dir = args.checkpoint_dir
    epoch_sample_num = args.epoch_sample_num

    # if checkpoint_dir is None:
    #     ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
    # else:
    #     ckpt_dir = checkpoint_dir

    logger.info("Initializing Tokenizer and Dataset.")

    tokenizer = AutoTokenizer.from_pretrained(bert_init)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    dataset = load_dataset(f"43shira43/{dataset}",cache_dir="/tmp/hf_cache")

    train, val, test = filter_and_tokenize(dataset, max_length)

    dataloaders = get_loaders(train, val, test, batch_size, use_weighted_sampler, epoch_sample_num, data_collator)
    logger.info("Successfully loaded all Dataloaders.")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    train_eval_loader = dataloaders["train_eval"]


    model = BertForTokenClassification.from_pretrained(
        bert_init,
        num_labels=5,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()}
    )

    model = model.to(gpu)

    optimizer = AdamW(model.parameters(), lr=bert_lr)

    trainer = Engine(partial(training_step, model=model, optimizer=optimizer))
    evaluator = Engine(partial(evaluation_step, model=model))

    # trainer = Engine(training_step)
    # evaluator = Engine(evaluation_step)

    pbar = ProgressBar()
    pbar.attach(trainer)

    # attach metrics
    # trainer: batch-level loss/acc aggregated per epoch
    Loss(cross_entropy, output_transform=lambda o: (o["y_pred"], o["y_true"])).attach(trainer, "loss")
    Accuracy(output_transform=lambda o: (o["y_pred"], o["y_true"])).attach(trainer, "acc")

    # evaluator: full-epoch metrics on val_loader and train_loader
    # metrics = {
    #     "loss": Loss(cross_entropy),
    #     "acc": Accuracy(),
    #     "precision": Precision(average=True),  # micro-avg over classes
    #     "recall": Recall(average=True),
    #     "f1": Fbeta(beta=1.0, average=True),
    #     "kappa": CohenKappa()
    # }
    #
    # for name, metric in metrics.items():
    #     metric.attach(evaluator, name)

    metrics = {
        "loss": Loss(cross_entropy),
        "acc": Accuracy(output_transform=cls_metric_transform, device=gpu),
        "precision": Precision(output_transform=cls_metric_transform, average=True, device=gpu),
        "recall": Recall(output_transform=cls_metric_transform, average=True, device=gpu),
        "f1": Fbeta(beta=1.0, output_transform=cls_metric_transform, average=True, device=gpu),
        "kappa": CohenKappa(output_transform=cls_metric_transform)  # CohenKappa moves to CPU/Numpy itself
    }
    for name, metric in metrics.items():
        metric.attach(evaluator, name)


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        global y_test_pred_results, y_test_true_results

        # Evaluate on all splits
        evaluator.run(train_eval_loader)
        train_metrics = dict(evaluator.state.metrics)  # or .copy()

        evaluator.run(val_loader)
        val_metrics = dict(evaluator.state.metrics)

        evaluator.run(test_loader)
        test_metrics = dict(evaluator.state.metrics)

        logger.info(
            f"[Epoch {trainer.state.epoch}] Train: acc={train_metrics['acc']:.4f} "
            f"loss={train_metrics['loss']:.4f} prec={train_metrics['precision']:.4f} recall={train_metrics['recall']:.4f} "
            f"f1={train_metrics['f1']:.4f} kappa={train_metrics['kappa']:.4f}"
        )

        logger.info(
            f"[Epoch {trainer.state.epoch}] Val:   acc={val_metrics['acc']:.4f} "
            f"loss={val_metrics['loss']:.4f} prec={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} kappa={val_metrics['kappa']:.4f}"
        )

        logger.info(
            f"[Epoch {trainer.state.epoch}] Test:  acc={test_metrics['acc']:.4f} "
            f"loss={test_metrics['loss']:.4f} prec={test_metrics['precision']:.4f} recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f} kappa={test_metrics['kappa']:.4f}"
        )


    logger.info("Starting trainer now...")
    trainer.run(train_loader, max_epochs=nb_epochs)
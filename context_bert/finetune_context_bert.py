import argparse
import logging
from typing import Optional, Literal
from ignite.contrib.handlers import ProgressBar
from ignite.metrics import Loss, Accuracy
from sklearn.metrics import cohen_kappa_score
from functools import partial
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
from helpers import get_loaders, parse_args, CohenKappa, training_step, evaluation_step


def collate_fn(batch):
    return tokenizer.pad(batch, return_tensors="pt")


def tokenization(example):
    text_col = 'sentence_text' if 'sentence_text' in example else 'text'
    return {"tokens": tokenizer(example[text_col],
                                add_special_tokens=False)["input_ids"]}



def get_max_context_sample(tokenized_sentences: list[list[int]],
                           sentence_idx: int,
                           max_length: int = 512,
                           max_context_sentences: Optional[int] = None,
                           context_type: Literal["prefix", "suffix"] = "prefix",
                           ) -> tuple[list[int], list[int]]:
    """Get the maximum context sample from a list of sentences.

    Args:
        tokenized_sentences: A list of input_ids of the sentences
        sentence_idx: The index of the sentence to get the context from.
        max_length: The maximum length of the context.
        max_context_sentences: Maximum number of context sentences to return.
        context_type: The type of context to get.

    Returns:
        The maximum context sample and the offset. text and buffer are lists with input_ids of the sentence
    """

    text = tokenized_sentences[sentence_idx]
    context = (
        # reverse the order of the prefix context
        tokenized_sentences[:sentence_idx][::-1]
        if context_type == "prefix"
        else tokenized_sentences[sentence_idx + 1:]
    )

    buffer = []
    for sentence in context[:max_context_sentences]:
        if len(buffer) + len(sentence) >= max_length:
            break

        if context_type == "suffix":
            buffer.extend(sentence)
        else:
            buffer = sentence + buffer

    return (text, buffer) if context_type == "suffix" else (buffer, text)


def build_context_dataset(ds,
                          max_context_sentences=2,
                          context_type: Literal["prefix", "suffix"] = "prefix"):
    tokens_list = ds["tokens"]
    labels      = ds["label"]
    rows = []
    for idx in range(len(ds)):
        ctx, tgt = get_max_context_sample(
            tokens_list,
            sentence_idx=idx,
            max_length=tokenizer.model_max_length,
            max_context_sentences=max_context_sentences,
            context_type=context_type,
        )
        feats = tokenizer.prepare_for_model(ctx, tgt, add_special_tokens=True)
        feats["labels"] = labels[idx]
        rows.append(feats)
    return Dataset.from_list(rows)


def filter_and_tokenize(data, max_context_sentences=2):
    """

    :param data: huggingface dataset with splits "train", "validation", "test", and columns "label" and "sentence_text" or "text"
    :param max_context_sentences: Maximum number of context sentences to return.
    :return: three instances datasets.Dataset with pytorch tensors as values to keys
    """
    # get split and keep only rows with label == 0, 1 or 2
    train_set = data["train"].filter(lambda example: example["label"] in [0, 1, 2])
    val_set = data["validation"].filter(lambda example: example["label"] in [0, 1, 2])
    test_set = data["test"].filter(lambda example: example["label"] in [0, 1, 2])

    train_set = train_set.map(tokenization, batched=True)
    val_set = val_set.map(tokenization, batched=True)
    test_set = test_set.map(tokenization, batched=True)

    train_dataset = build_context_dataset(train_set, max_context_sentences, "prefix")
    val_dataset = build_context_dataset(val_set, max_context_sentences, "prefix")
    test_dataset = build_context_dataset(test_set, max_context_sentences, "prefix")

    # turn to torch tensors for dataloader
    cols = ["input_ids", "token_type_ids", "attention_mask", "labels"]
    for ds in (train_dataset, val_dataset, test_dataset):
        ds.set_format(type="torch", columns=cols)

    return train_dataset, val_dataset, test_dataset



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
    max_context = args.max_context_sentences
    # is_save_model = args.is_save_model
    dataset = args.dataset
    bert_init = args.bert_init
    # checkpoint_dir = args.checkpoint_dir
    epoch_sample_num = args.epoch_sample_num

    # if checkpoint_dir is None:
    #     ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset)
    # else:
    #     ckpt_dir = checkpoint_dir

    logger.info("Initializing Tokenizer adn Dataset.")

    tokenizer = AutoTokenizer.from_pretrained(bert_init)

    dataset = load_dataset(f"43shira43/{dataset}",cache_dir="/tmp/hf_cache")

    train, val, test = filter_and_tokenize(dataset, max_context)

    logger.info("Loading Dataloaders now...")
    dataloaders = get_loaders(train, val, test, batch_size, use_weighted_sampler, epoch_sample_num, collate_fn)
    logger.info("Successfully loaded all Dataloaders.")

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    train_eval_loader = dataloaders["train_eval"]



    # TODO testen ob evtl für token classification bessere ergebnisse erzielt werden ?
    # TODO roberta does not have token_type_ids
    model = BertForSequenceClassification.from_pretrained(
        bert_init,
        num_labels=3,
        problem_type="single_label_classification"
    )
    model = model.to(gpu)

    optimizer = AdamW(model.parameters(), lr=bert_lr)

    trainer = Engine(partial(training_step, model=model, optimizer=optimizer))
    evaluator = Engine(partial(evaluation_step, model=model))

    pbar = ProgressBar()
    pbar.attach(trainer)

    # attach metrics
    # trainer: batch-level loss/acc aggregated per epoch
    Loss(cross_entropy, output_transform=lambda o: (o["y_pred"], o["y_true"])).attach(trainer, "loss")
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

        # # Save best checkpoint (tracked via val κappa)
        # if not hasattr(log_training_results, "best_val_kappa"):
        #     log_training_results.best_val_kappa = 0
        #
        # if val_metrics['kappa'] > log_training_results.best_val_kappa:
        #     log_training_results.best_val_kappa = val_metrics['kappa']
        #     logger.info("New best val kappa -> saving checkpoint")
        #
        #     # Convert numeric labels to readable labels if you want (optional)
        #     # lable_list_path = "./data/corpus/we_labels.txt".replace("we", dataset)
        #     # labels_dic = get_predict_lable_dic(lable_list_path)
        #     # y_test_pred_results = [labels_dic[p] for p in y_test_pred_results]
        #     # y_test_true_results = [labels_dic[t] for t in y_test_true_results]
        #
        #     prediction_data = pd.DataFrame({
        #         'label_': y_test_true_results,
        #         'label_preds': y_test_pred_results
        #     })
        #
        #     # Save test predictions
        #     os.makedirs(ckpt_dir, exist_ok=True)
        #     predicted_test_data_path = os.path.join(ckpt_dir, 'predicted_test_data.xlsx')
        #     prediction_data.to_excel(predicted_test_data_path, index=False)
        #
        #     # Save model checkpoint
        #     if args.is_save_model == "save":
        #         th.save({
        #             'bert_model': model.bert.state_dict(),
        #             'classifier': model.classifier.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'epoch': trainer.state.epoch,
        #         }, os.path.join(ckpt_dir, 'checkpoint.pth'))


    # @trainer.on(Events.EPOCH_COMPLETED)
    # def log_train_metrics(engine):
    #     m = engine.state.metrics
    #     print(f"[Train] Epoch {engine.state.epoch:02d} | "
    #           f"loss={m['loss']:.4f} | acc={m['acc']:.4f}")
    #     logger.info(f"[Train] Epoch {engine.state.epoch:02d} | "
    #                 f"loss={m['loss']:.4f} | acc={m['acc']:.4f}")
    #
    #
    # @trainer.on(Events.EPOCH_COMPLETED)
    # def validate(engine):
    #     evaluator.run(val_loader)
    #     m = evaluator.state.metrics
    #     print(f"[Val]   Epoch {engine.state.epoch:02d} | "
    #           f"loss={m['loss']:.4f} | "
    #           f"acc={m['acc']:.4f} | "
    #           f"P={m['precision']:.4f} R={m['recall']:.4f} "
    #           f"F1={m['f1']:.4f} κ={m['kappa']:.4f}")
    #     logger.info(f"[Val]   Epoch {engine.state.epoch:02d} | "
    #                 f"loss={m['loss']:.4f} | "
    #                 f"acc={m['acc']:.4f} | "
    #                 f"P={m['precision']:.4f} R={m['recall']:.4f} "
    #                 f"F1={m['f1']:.4f} κ={m['kappa']:.4f}")
    #

    logger.info("Starting trainer now...")
    trainer.run(train_loader, max_epochs=nb_epochs)

    # ─── after training finishes ───────────────────────────────────────────────────
    logger.info("Further Model Evaluation after training has finished.")
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
        logger.info(f"{tok:>10}  seg={ttype}  attn={score:.3f}")

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

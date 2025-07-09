import warnings
from typing import TypedDict, Union
import evaluate
import numpy as np
from datasets import Dataset
from numpy.typing import NDArray
from transformers.trainer_utils import EvalPrediction
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding



warnings.filterwarnings("ignore", message=r".*Please note that with a fast tokenizer.*")
warnings.filterwarnings(
    "ignore",
    message=r".*Using the `WANDB_DISABLED` environment variable is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Was asked to gather along dimension \d+, but all input tensors were scalars.*",
)


def compute_scores(
    scores: NDArray, threshold: float, labels: NDArray, suffix=""
) -> dict[str, float]:
    evaluate_f1 = evaluate.load("f1")
    evaluate_acc = evaluate.load("accuracy")
    evaluate_roc_auc = evaluate.load("roc_auc")

    preds = (scores >= threshold).astype(int)
    f1_score_weighted = evaluate_f1.compute(
        predictions=preds, references=labels, average="weighted"
    )
    acc_score = evaluate_acc.compute(predictions=preds, references=labels)
    roc_auc_score = evaluate_roc_auc.compute(
        prediction_scores=scores, references=labels
    )
    computed_scores = {
        f"f1_weighted{suffix}": float(f1_score_weighted["f1"]),  # type: ignore
        f"accuracy{suffix}": float(acc_score["accuracy"]),  # type: ignore
        f"roc_auc{suffix}": float(roc_auc_score["roc_auc"]),  # type: ignore
    }

    f1_score_each = evaluate_f1.compute(
        predictions=preds, references=labels, average=None
    )
    try:
        computed_scores |= {
            f"f1_human{suffix}": float(f1_score_each["f1"][0]),  # type: ignore
            f"f1_ai{suffix}": float(f1_score_each["f1"][1]),  # type: ignore
        }
    except TypeError:
        UserWarning(
            "evaluate(f1) with average=None returned a single value, "
            "indicating only one class is present in the dataset."
        )
        if labels[0] == 0:
            computed_scores |= {
                f"f1_human{suffix}": float(f1_score_each["f1"]),  # type: ignore
            }
        else:
            computed_scores |= {
                f"f1_ai{suffix}": float(f1_score_each["f1"]),  # type: ignore
            }

    return computed_scores


class EvaluationMetrics(TypedDict):
    f1_weighted: float
    accuracy: float
    roc_auc: float
    f1_human: float
    f1_ai: float
    n_samples: int


class Balanced(EvaluationMetrics):
    f1_weighted_median: float
    accuracy_median: float
    roc_auc_median: float
    f1_human_median: float
    f1_ai_median: float
    threshold_median: float


class Unbalanced(EvaluationMetrics):
    f1_weighted_mean: float
    accuracy_mean: float
    roc_auc_mean: float
    f1_human_mean: float
    f1_ai_mean: float
    threshold_mean: float
    n_samples_human: int
    n_samples_ai: int


def compute_metrics(
    eval_pred: EvalPrediction,
) -> Union[EvaluationMetrics, Balanced, Unbalanced]:
    """Calculated weigthed and class-specific F1 scores, accuracy, ROC AUC, and class distribution for a threshold of 0.5.
    In addition, we compute best-guess thresholds based on the dataset balance:
    - If the dataset is balanced, we use the median of all scores as threshold.
    - If the dataset is unbalanced, we use the midpoint between the means of the two classes as threshold.
    - If only one class is present, no additional threshold is considered.

    Args:
        eval_pred: Tuple of logits and labels from the model's predictions.

    Returns:
        EvaluationMetrics: A dictionary containing calcualted metrics.
    """
    logits, labels = eval_pred  # type: ignore

    labels: NDArray = np.array(labels)
    scores: NDArray = 1 / (1 + np.exp(-np.array(logits)))

    n_samples_human = int(np.sum(labels == 0))
    n_samples_ai = int(np.sum(labels == 1))

    metrics = EvaluationMetrics(
        **compute_scores(scores, 0.5, labels),
        n_samples=len(labels),
    )

    if n_samples_human == n_samples_ai:
        # dataset is balanced, use the median of all scores as threshold
        threshold = float(np.median(scores))
        metrics_median = compute_scores(scores, threshold, labels, "_median")
        return Balanced(**metrics, **metrics_median, threshold_median=threshold)
    elif n_samples_human > 0 < n_samples_ai:
        # dataset is unbalanced, use the midpoint between the means of the two class distributions as threshold
        threshold = float(scores[labels == 0].mean() + scores[labels == 1].mean()) / 2
        metrics_mean = compute_scores(scores, threshold, labels, "_mean")
        return Unbalanced(
            **metrics,
            **metrics_mean,
            threshold_mean=threshold,
            n_samples_human=n_samples_human,
            n_samples_ai=n_samples_ai,
        )
    else:
        # only one class is present
        # TODO?
        pass

    return metrics


class PredictionResults(TypedDict):
    prediction: list[float]


class DetectorABC(ABC):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        device: Union[str, torch.device] = ("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.tokenizer = tokenizer

    @abstractmethod
    def tokenize(self, texts: list[str]) -> BatchEncoding: ...

    @abstractmethod
    def process(self, inputs: dict) -> PredictionResults: ...


def run_detector_tokenized(detector: DetectorABC, dataset: Dataset, batch_size=32):
    labels = []
    predictions = []
    for batch in dataset.batch(batch_size):
        labels.extend(batch["labels"])  # type: ignore
        predictions.extend(detector.process(batch)["prediction"])  # type: ignore

    return compute_metrics((np.array(predictions), np.array(labels)))  # type: ignore


def run_detector(detector: DetectorABC, dataset: Dataset, batch_size=32):
    """Sorting the samples by length (in number of tokens) enables efficient batching
    as batches of similar length have reduced overhead.

    Note:
        Requires `detector.tokenize` to return "length" field!
    """
    dataset = dataset.map(
        detector.tokenize,
        input_columns=["text"],
        batched=True,
        batch_size=1024,
        desc="Tokenizing",
    ).sort("length")

    return run_detector_tokenized(detector, dataset, batch_size=batch_size)



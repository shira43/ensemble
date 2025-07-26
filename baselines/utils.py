import warnings
from typing import TypedDict, Union
import evaluate
import numpy as np
from datasets import Dataset
from numpy.typing import NDArray
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
from tqdm import tqdm
from transformers.trainer_utils import EvalPrediction
from abc import ABC, abstractmethod
import torch
import numba as nb
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding
from ignite.metrics import Metric
import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(message)s"
)

warnings.filterwarnings("ignore", message=r".*Please note that with a fast tokenizer.*")
warnings.filterwarnings(
    "ignore",
    message=r".*Using the `WANDB_DISABLED` environment variable is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    message=r".*Was asked to gather along dimension \d+, but all input tensors were scalars.*",
)


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






@nb.jit(nopython=True)
def sign(x):
    return -1 if x < 0 else 1


Threshold = float
FPR = float



@nb.jit(nopython=True)
def find_threshold_for_fpr(
    y_scores: NDArray,
    target_fpr: float = 0.05,
    epsilon: float = 0.0005,
    greater: bool = True,
) -> Threshold:
    """

    Modified from: RAID, Dugan et al. 2024.

    Source:
        https://github.com/liamdugan/raid/blob/main/raid/evaluate.py#L18

    Args:
        y_scores (list[float] | NDArray): The predicted scores for the human-written texts.
        target_fpr (float): The target false-positive rate to achieve.
        epsilon (float): The acceptable error margin for the false-positive rate.
        greater (bool, optional): If true, a score greater than the threshold is considered a positive prediction.

    Returns:
        tuple[Threshold, FPR]: a tuple of floats representing the threshold and the corresponding false-positive rate.
    """
    # Initialize the list of all found thresholds and FPRs
    prev_dist = np.nan
    step_size = 0.5
    found_threshold_list = []

    threshold: Threshold = y_scores.mean()
    for _ in range(50):
        # Calculate true predictions as `great-than-or-equal-to current threshold`
        # and flip it using XOR with `not greater` if needed
        y_pred = (y_scores >= threshold) ^ (not greater)

        # Ground truth is 0 for human-written texts, so all predictions of 1 are false positives
        fpr = float(np.mean((y_pred) == 1))

        # If we reached the target FPR, return the threshold
        if abs(fpr - target_fpr) <= epsilon:
            return threshold

        # Save the computed values to the found_threshold_list
        found_threshold_list.append((threshold, fpr))

        # Compute distance
        dist = target_fpr - fpr

        # If dist and prev_dist are different signs then swap
        # sign of step size and cut in half
        if prev_dist != np.nan and sign(dist) != sign(prev_dist):
            step_size *= -0.5
        # Otherwise if we're going the wrong direction, then just swap sign of step
        elif prev_dist != np.nan and abs(dist) - abs(prev_dist) > 0.01:
            step_size *= -1

        # Step the threshold value and save prev_dist
        threshold += step_size
        prev_dist = dist

    # Compute diffs for all thresholds found during search
    # (Exclude all thresholds for which the true fpr is 0)
    diffs = [(target_fpr - fpr, t) for t, fpr in found_threshold_list if fpr > 0.0]

    # If there are positive numbers in the list, pick threshold for smallest pos number
    # Otherwise pick the threshold for the negative diff value closest to 0
    pos_diffs = [(d, t) for d, t in diffs if d >= 0]
    if len(pos_diffs) > 0:
        threshold = min(pos_diffs)[1]
    else:
        threshold = max(diffs)[1]

    return threshold


def calculate_metrics(
    y_true: NDArray,
    y_scores: NDArray,
    threshold: float,
    suffix="",
    greater: bool = True,
) -> dict[str, float]:
    y_preds = (
        (y_scores >= threshold).astype(int)
        if greater
        else (y_scores <= threshold).astype(int)
    )
    precision, recall, f1_score, _support = precision_recall_fscore_support(
        y_true, y_preds, average="weighted"
    )
    accuracy = np.mean(y_true == y_preds)
    fpr = np.mean(y_preds[y_true == 0] == 1)
    tpr = np.mean(y_preds[y_true == 1] == 1)

    if y_scores.min() < 0 or y_scores.max() > 1:
        norm_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        norm_scores = y_scores
    if not greater:
        norm_scores = 1 - norm_scores
    #roc_auc = roc_auc_score(y_true, y_preds, average="weighted")

    calculated_metrics = {
        f"f1_score{suffix}": float(f1_score),  # type: ignore
        f"precision{suffix}": float(precision),  # type: ignore
        f"recall{suffix}": float(recall),  # type: ignore
        f"accuracy{suffix}": float(accuracy),  # type: ignore
        "kappa": CohenKappa(),
        #f"roc_auc{suffix}": float(roc_auc),  # type: ignore
        f"fpr{suffix}": float(fpr),  # type: ignore
        f"tpr{suffix}": float(tpr),  # type: ignore

    }

    _precision_each, _recall_each, f1_score_each, _support_each = (
        precision_recall_fscore_support(y_true, y_preds, average=None)
    )
    try:
        calculated_metrics |= {
            f"f1_human{suffix}": float(f1_score_each[0]),  # type: ignore
            f"f1_ai{suffix}": float(f1_score_each[1]),  # type: ignore
        }
    except Exception:
        UserWarning("precision_recall_fscore_support(average=None) raised an exception")

    return calculated_metrics


def compute_metrics(
    eval_pred: EvalPrediction,
    threshold: float = 0.5,
    sigmoid: bool = True,
    greater: bool = True,
) -> dict[str, Union[float, int]]:
    """Calculated weigthed and class-specific F1 scores, accuracy, ROC AUC, and class distribution for a threshold of 0.5.
    In addition, we compute best-guess thresholds based on the dataset balance:
    - If the dataset is balanced, we use the median of all scores as threshold.
    - If the dataset is unbalanced, we use the midpoint between the means of the two classes as threshold.
    - If only one class is present, no additional threshold is considered.

    Args:
        eval_pred: Tuple of logits and labels from the model's predictions.
        threshold: The threshold to use for classification.
        sigmoid: Whether to apply the sigmoid function to the logits.
        greater: Whether a higher score indicates a positive class (True) or a lower score (False).

    Returns:
        EvaluationMetrics: A dictionary containing calcualted metrics.
    """
    logits, y_true = eval_pred  # type: ignore

    y_true: NDArray = np.array(y_true)
    y_scores: NDArray = np.array(logits)
    if sigmoid:
        # Convert logits to probabilities using the sigmoid function
        y_scores = 1 / (1 + np.exp(-y_scores))

    n_samples_human = int(np.sum(y_true == 0))
    n_samples_ai = int(np.sum(y_true == 1))

    metrics = {"n_samples": len(y_true)}
    metrics |= calculate_metrics(y_true, y_scores, threshold, greater=greater)

    if n_samples_human == n_samples_ai:
        # dataset is balanced, use the median of all scores as threshold
        threshold_median = float(np.median(y_scores))
        metrics_median = calculate_metrics(
            y_true, y_scores, threshold_median, "_median", greater=greater
        )
        metrics |= metrics_median | {"threshold_median": threshold_median}

    # Use the midpoint between the means of the two class distributions as threshold
    # works if the dataset is unbalanced
    threshold_mean = (
        float(y_scores[y_true == 0].mean() + y_scores[y_true == 1].mean()) / 2
    )
    metrics_mean = calculate_metrics(
        y_true, y_scores, threshold_mean, "_mean", greater=greater
    )
    metrics |= metrics_mean | {"threshold_mean": threshold_mean}

    # Use the midpoint between the means of the two class distributions as threshold
    # works if the dataset is unbalanced
    threshold_fpr = find_threshold_for_fpr(
        y_scores[y_true == 0], target_fpr=0.05, epsilon=0.0005, greater=greater
    )
    metrics_fpr = calculate_metrics(
        y_true, y_scores, threshold_fpr, "_fpr", greater=greater
    )
    metrics |= metrics_fpr | {"threshold_fpr": threshold_fpr}

    if n_samples_human > 0 < n_samples_ai:
        metrics |= dict(
            n_samples_human=n_samples_human,
            n_samples_ai=n_samples_ai,
        )

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

#
# def run_detector_tokenized(detector: DetectorABC, dataset: Dataset, batch_size=32):
#     labels = []
#     predictions = []
#     for i in range(0, len(dataset), batch_size):
#         batch = dataset[i: i + batch_size]
#         labels.extend(batch["label"])  # type: ignore
#         predictions.extend(detector.process(batch)["prediction"])  # type: ignore
#
#     logging.info("Returning logits....")
#     return predictions
#     # logging.info("Starting compute_metrics...")
#     # return compute_metrics((np.array(predictions), np.array(labels)))  # type: ignore
#
#
# def run_detector(detector: DetectorABC, dataset: Dataset, batch_size=32):
#     """Sorting the samples by length (in number of tokens) enables efficient batching
#     as batches of similar length have reduced overhead.
#
#     Note:
#         Requires `detector.tokenize` to return "length" field!
#     """
#     dataset = dataset.map(
#         detector.tokenize,
#         input_columns=["text"],
#         batched=True,
#         batch_size=1024,
#         desc="Tokenizing",
#     ).sort("length")
#
#     logging.info("Starting run_detector_tokenized...")
#     return run_detector_tokenized(detector, dataset, batch_size=batch_size)



def run_detector_tokenized(
    detector: DetectorABC,
    dataset: Dataset,
    batch_size=32,
    threshold: float = 0.5,
    sigmoid: bool = True,
    greater: bool = True,
):
    labels = []
    predictions = []
    for batch in tqdm(dataset.batch(batch_size), desc="Processing Batches"):
        labels.extend(batch["label"])  # type: ignore
        predictions.extend(detector.process(batch)["prediction"])  # type: ignore

    return compute_metrics(
        (np.array(predictions), np.array(labels)),
        threshold=threshold,
        sigmoid=sigmoid,
        greater=greater,
    )  # type: ignore


def run_detector(
    detector: DetectorABC,
    dataset: Dataset,
    batch_size=32,
    threshold: float = 0.5,
    sigmoid: bool = True,
    greater: bool = True,
):
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

    return run_detector_tokenized(
        detector,
        dataset,
        batch_size=batch_size,
        threshold=threshold,
        sigmoid=sigmoid,
        greater=greater,
    )
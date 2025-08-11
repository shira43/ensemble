#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Threshold learning for 3-class Binoculars (human/mixed/AI)
# - Adaptive grid over class-conditional quantile ranges
# - Practical constraints (human FPR cap, min mixed share, min gap)
# - Optional bootstrap stability
# - Clean evaluation on a held-out test split

import os
import gc
import json
import math
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)

# ------------------------------------------------------------
# Your Binoculars class (expects .tokenizer, .observer_model, .performer_model, .predict)
# If your file is named binoculars.py and contains class Binoculars, this import works:
from binoculars import Binoculars
# ------------------------------------------------------------

# ----------------------- CONFIG -----------------------------
@dataclass
class CONFIG:
    hf_ds_name_val: str = "43shira43/coauthor-extended-np"
    hf_ds_split_val: str = "validation"
    hf_ds_name_test: str = "43shira43/coauthor-extended-np"
    hf_ds_split_test: str = "test"

    # batching & tokenization
    batch_size: int = 16
    max_token_observed: int = 512  # will be picked from detector

    # grid search params (numbers of points)
    n_hi: int = 101
    n_lo: int = 101
    lo_q: float = 0.05
    hi_q: float = 0.95
    min_gap: float = 1e-4

    # constraints (policy knobs)
    max_human_fpr: float = 0.05      # <= 5% humans predicted as AI
    min_mixed_share: float = 0.05    # >= 5% predictions should be mixed

    # bootstrap stability (optional)
    use_bootstrap: bool = False
    n_bootstrap: int = 200
    random_seed: int = 13

    # output
    plot_path: str = "score_hist_extended_new.png"
    save_thresholds_json: str = "binoculars_thresholds.json"


CFG = CONFIG()
np.random.seed(CFG.random_seed)
rng = np.random.default_rng(CFG.random_seed)
torch.manual_seed(CFG.random_seed)


# ----------------------- HELPERS -----------------------------
def score_to_label(score_arr: np.ndarray, t_hi: float, t_lo: float) -> np.ndarray:
    """
    Map score -> {0,1,2} with two thresholds.
    Convention: higher score = more human-like.
      score >= t_hi -> human (0)
      score <= t_lo -> ai    (1)
      else           -> mixed (2)
    """
    lab = np.full_like(score_arr, 2, dtype=int)
    lab[score_arr >= t_hi] = 0
    lab[score_arr <= t_lo] = 1
    return lab


def human_fpr(scores: np.ndarray, labels: np.ndarray, preds: np.ndarray) -> float:
    """Among true humans (0), fraction predicted AI (1)."""
    is_h = labels == 0
    if not np.any(is_h):
        return 0.0
    return float(np.mean(preds[is_h] == 1))


def make_grids(scores: np.ndarray, labels: np.ndarray,
               n_hi: int, n_lo: int, lo_q: float, hi_q: float):
    """
    Build adaptive grids for the two boundaries using robust quantile ranges
    over (human ∪ mixed) for t_hi and (mixed ∪ ai) for t_lo.
    """
    H = scores[labels == 0]
    A = scores[labels == 1]
    M = scores[labels == 2]
    # If any class is missing in val, fall back to global quantiles
    hm = np.concatenate([H, M]) if H.size and M.size else scores
    ma = np.concatenate([M, A]) if M.size and A.size else scores

    hi_min = float(np.quantile(hm, lo_q))
    hi_max = float(np.quantile(hm, hi_q))
    lo_min = float(np.quantile(ma, lo_q))
    lo_max = float(np.quantile(ma, hi_q))

    # handle degenerate ranges
    if not math.isfinite(hi_min) or not math.isfinite(hi_max) or hi_min == hi_max:
        hi_min, hi_max = float(np.quantile(scores, lo_q)), float(np.quantile(scores, hi_q))
    if not math.isfinite(lo_min) or not math.isfinite(lo_max) or lo_min == lo_max:
        lo_min, lo_max = float(np.quantile(scores, lo_q)), float(np.quantile(scores, hi_q))

    HI_GRID = np.linspace(hi_min, hi_max, n_hi)
    LO_GRID = np.linspace(lo_min, lo_max, n_lo)
    return HI_GRID, LO_GRID


def objective_macroF1_with_constraints(
    scores: np.ndarray,
    labels: np.ndarray,
    t_hi: float,
    t_lo: float,
    max_human_fpr: float,
    min_mixed_share: float,
    min_gap: float,
) -> float:
    """Return macro-F1 if constraints satisfied, else -inf."""
    if not (t_lo + min_gap < t_hi):
        return -np.inf
    preds = score_to_label(scores, t_hi, t_lo)

    # constraints
    h_fpr = human_fpr(scores, labels, preds)
    if h_fpr > max_human_fpr:
        return -np.inf

    mixed_share = float(np.mean(preds == 2))
    if mixed_share < min_mixed_share:
        return -np.inf

    return float(f1_score(labels, preds, average="macro", zero_division=0))


def bootstrap_macroF1(
    scores: np.ndarray,
    labels: np.ndarray,
    t_hi: float,
    t_lo: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Mean and std of macro-F1 via bootstrap."""
    n = len(scores)
    vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        preds = score_to_label(scores[idx], t_hi, t_lo)
        vals.append(f1_score(labels[idx], preds, average="macro", zero_division=0))
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std())


def learn_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    n_hi: int,
    n_lo: int,
    lo_q: float,
    hi_q: float,
    max_human_fpr: float,
    min_mixed_share: float,
    min_gap: float,
    use_bootstrap: bool,
    n_bootstrap: int,
) -> tuple[float, float, dict]:
    """
    Grid search for (t_hi, t_lo) maximizing macro-F1 under constraints.
    If use_bootstrap=True, choose by mean bootstrap F1 and break ties
    by lower std.
    """
    HI_GRID, LO_GRID = make_grids(scores, labels, n_hi, n_lo, lo_q, hi_q)

    best = -np.inf
    best_std = np.inf
    best_hi = None
    best_lo = None

    for t_hi in HI_GRID:
        for t_lo in LO_GRID:
            val = objective_macroF1_with_constraints(
                scores, labels, t_hi, t_lo,
                max_human_fpr=max_human_fpr,
                min_mixed_share=min_mixed_share,
                min_gap=min_gap,
            )
            if val == -np.inf:
                continue

            if use_bootstrap:
                meanF1, stdF1 = bootstrap_macroF1(scores, labels, t_hi, t_lo, n_bootstrap, rng)
                # prefer higher mean; tie-breaker: lower std
                if (meanF1 > best) or (np.isclose(meanF1, best) and stdF1 < best_std):
                    best, best_std, best_hi, best_lo = meanF1, stdF1, t_hi, t_lo
            else:
                if val > best:
                    best, best_hi, best_lo = val, t_hi, t_lo

    if best_hi is None or best_lo is None:
        raise RuntimeError("Failed to find thresholds under given constraints. "
                           "Try relaxing max_human_fpr/min_mixed_share or widening quantile ranges.")

    meta = dict(
        grid_hi=int(n_hi),
        grid_lo=int(n_lo),
        lo_q=float(lo_q),
        hi_q=float(hi_q),
        objective="macroF1_with_constraints",
        best_score=float(best),
        bootstrap_used=bool(use_bootstrap),
        bootstrap_std=float(best_std) if use_bootstrap else None,
        constraints=dict(
            max_human_fpr=float(max_human_fpr),
            min_mixed_share=float(min_mixed_share),
            min_gap=float(min_gap),
        ),
    )
    return float(best_hi), float(best_lo), meta


def collect_scores(dataset, detector: Binoculars, batch_size=16) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Binoculars and collect raw scores + labels.
    Assumes dataset has 'text' and 'label' fields (0=human,1=ai,2=mixed).
    """
    labels, scores = [], []
    for batch in tqdm(dataset.batch(batch_size), desc="Binoculars"):
        texts = batch["text"]
        labels.extend(batch["label"])
        enc = detector.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=detector.max_token_observed,
            return_tensors="pt",
        ).to(detector.observer_model.device)
        # bypass .process (which expects token dict with input_ids/attention_mask)
        scores.extend(detector.predict(enc))
    return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)


def plot_hist(scores: np.ndarray, labels: np.ndarray, t_hi: float, t_lo: float, out_path: str):
    df = pd.DataFrame({"score": scores, "label": labels})
    labels_map = {0: "human", 1: "ai", 2: "mixed"}
    plt.figure(figsize=(7, 4))
    for lbl, name in labels_map.items():
        arr = df.loc[df.label == lbl, "score"].values
        if arr.size:
            plt.hist(arr, bins=60, density=True, alpha=0.5, label=name)
    plt.axvline(t_hi, linestyle="--", linewidth=2)
    plt.axvline(t_lo, linestyle="--", linewidth=2)
    plt.xlabel("Binoculars score")
    plt.ylabel("density")
    plt.title("Score distribution with learned thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ----------------------- MAIN -----------------------------
def main():
    # Load data
    val_ds = load_dataset(CFG.hf_ds_name_val, split=CFG.hf_ds_split_val)
    test_ds = load_dataset(CFG.hf_ds_name_test, split=CFG.hf_ds_split_test)

    # If needed:
    # val_ds  = val_ds.rename_columns({"sentence_text": "text"}).filter(lambda ex: ex["label"] in [0,1,2])
    # test_ds = test_ds.rename_columns({"sentence_text": "text"}).filter(lambda ex: ex["label"] in [0,1,2])

    # Build detector
    det = Binoculars()  # customize constructor if you use different backbones

    try:
        # Collect scores
        val_scores,  val_labels  = collect_scores(val_ds,  det, batch_size=CFG.batch_size)
        test_scores, test_labels = collect_scores(test_ds, det, batch_size=CFG.batch_size)

        # Learn thresholds on validation
        t_hi, t_lo, meta = learn_thresholds(
            val_scores, val_labels,
            n_hi=CFG.n_hi, n_lo=CFG.n_lo,
            lo_q=CFG.lo_q, hi_q=CFG.hi_q,
            max_human_fpr=CFG.max_human_fpr,
            min_mixed_share=CFG.min_mixed_share,
            min_gap=CFG.min_gap,
            use_bootstrap=CFG.use_bootstrap,
            n_bootstrap=CFG.n_bootstrap,
        )

        print(f"[VAL] best objective={meta['best_score']:.4f}  at t_hi={t_hi:.6f}  t_lo={t_lo:.6f}")
        if CFG.use_bootstrap:
            print(f"[VAL] bootstrap std={meta['bootstrap_std']:.4f}")

        # Save thresholds & meta
        with open(CFG.save_thresholds_json, "w", encoding="utf-8") as f:
            json.dump(
                dict(t_hi=t_hi, t_lo=t_lo, meta=meta),
                f, ensure_ascii=False, indent=2
            )
        print(f"[INFO] thresholds saved to {CFG.save_thresholds_json}")

        # Evaluate on TEST
        test_pred = score_to_label(test_scores, t_hi, t_lo)
        print("\n[TEST] classification report")
        print(classification_report(test_labels, test_pred, digits=3))
        print("[TEST] Cohen κ:", cohen_kappa_score(test_labels, test_pred))
        print("[TEST] Confusion matrix (rows=true [0,1,2], cols=pred [0,1,2])")
        print(confusion_matrix(test_labels, test_pred, labels=[0, 1, 2]))

        # Diagnostics
        h_fpr = human_fpr(test_scores, test_labels, test_pred)
        mix_share = float(np.mean(test_pred == 2))
        print(f"[TEST] human FPR={h_fpr:.4f} | mixed share={mix_share:.4f}")

        # Plot
        plot_hist(test_scores, test_labels, t_hi, t_lo, CFG.plot_path)
        print(f"[INFO] plot written to {CFG.plot_path}")

    finally:
        # Tidy GPU
        try:
            det.observer_model.to("cpu")
            det.performer_model.to("cpu")
        except Exception:
            pass
        del det
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()

from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd, torch, gc
from tqdm import tqdm
from binoculars import Binoculars          # your class

VAL_DS = load_dataset("43shira43/coauthor-extended-np", split="validation") \
           # .rename_columns({"sentence_text": "text"})            \
           # .filter(lambda ex: ex["label"] in [0, 1, 2])

TEST_DS = load_dataset("43shira43/coauthor-extended-np", split="test") \
           # .rename_columns({"sentence_text": "text"})            \
           # .filter(lambda ex: ex["label"] in [0, 1, 2])

def collect_scores(dataset, detector, batch_size=16):
    labels, scores = [], []

    for batch in tqdm(dataset.batch(batch_size), desc="Binoculars"):
        # 1) grab raw texts
        texts  = batch["text"]
        labels.extend(batch["label"])

        # 2) tokenize -> tensors on the observer model's device
        enc = detector.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=detector.max_token_observed,
            return_tensors="pt"
        ).to(detector.observer_model.device)

        # 3) call detector.predict (bypasses .process which needs tokenized dict)
        scores.extend(detector.predict(enc))

    return np.array(scores), np.array(labels)


det = Binoculars()

val_scores,  val_labels  = collect_scores(VAL_DS,  det)
test_scores, test_labels = collect_scores(TEST_DS, det)

# tidy GPU ↓
det.observer_model.to("cpu"); det.performer_model.to("cpu")
del det; gc.collect(); torch.cuda.empty_cache()


from sklearn.metrics import f1_score
import itertools

def score_to_label(score_arr, t_hi, t_lo):
    lab = np.full_like(score_arr, 2, dtype=int)
    lab[score_arr >= t_hi] = 0
    lab[score_arr <= t_lo] = 1
    return lab

# search ranges (adapt to your score distribution!)
HI_GRID = np.linspace(0.80, 1.05, 51)      # upper boundary candidates
LO_GRID = np.linspace(0.60, 0.95, 71)      # lower boundary candidates

best_f1, best_hi, best_lo = -1, None, None
for t_hi, t_lo in itertools.product(HI_GRID, LO_GRID):
    if t_lo >= t_hi:                 # must stay ordered
        continue
    preds = score_to_label(val_scores, t_hi, t_lo)
    f1 = f1_score(val_labels, preds, average="macro", zero_division=0)
    if f1 > best_f1:
        best_f1, best_hi, best_lo = f1, t_hi, t_lo

print(f"best macro-F1={best_f1:.3f} at  t_hi={best_hi:.3f}, t_lo={best_lo:.3f}")


from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

test_pred = score_to_label(test_scores, best_hi, best_lo)

print(classification_report(test_labels, test_pred, digits=3))
print("Cohen κ:", cohen_kappa_score(test_labels, test_pred))
print("Confusion matrix\n", confusion_matrix(test_labels, test_pred, labels=[0,1,2]))



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



df = pd.DataFrame({"score": test_scores, "label": test_labels})
labels_map = {0: "human", 1: "ai", 2: "mixed"}

plt.figure(figsize=(7, 4))
for lbl, name in labels_map.items():
    plt.hist(df[df.label == lbl].score,
             bins=60, density=True, alpha=0.5, label=name)

plt.axvline(best_hi, linestyle="--", linewidth=2)
plt.axvline(best_lo, linestyle="--", linewidth=2)
plt.xlabel("Binoculars score")
plt.ylabel("density")
plt.title("Score distribution with learnt thresholds")
plt.legend()
plt.tight_layout()

#save
plt.savefig("score_hist_extended.png", dpi=150)
plt.close()
print("Plot written to score_hist_extended.png")
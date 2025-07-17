# sliding_dataset.py
import itertools, torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

class SlidingWindowDataset(Dataset):
    """
    Build k-sentence windows around every sentence.
    df must have columns ['doc_id', 'text', 'label'].
    """
    def __init__(self, df, tokenizer: PreTrainedTokenizer,
                 k: int = 2, max_len: int = 256):
        self.tokenizer = tokenizer
        self.k, self.max_len = k, max_len

        # group sents per document
        self.docs = {}
        for _, row in df.iterrows():
            self.docs.setdefault(row.doc_id, []).append((row.text, row.label))

        self.index = list(itertools.chain.from_iterable(
            [(doc_id, i) for i in range(len(sents))]
            for doc_id, sents in self.docs.items()
        ))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        doc_id, sent_idx = self.index[idx]
        sents_labels = self.docs[doc_id]

        i0 = max(0, sent_idx - self.k)
        i1 = min(len(sents_labels), sent_idx + self.k + 1)
        window = " {} ".format(self.tokenizer.sep_token).join(
                    s for s, _ in sents_labels[i0:i1])

        enc = self.tokenizer(window,
                             truncation=True,
                             max_length=self.max_len,
                             padding="max_length",
                             return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(sents_labels[sent_idx][1],
                                           dtype=torch.long)
        }

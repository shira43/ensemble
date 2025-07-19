from torch.utils.data import Dataset
from typing import Literal, Optional


class BertContextDataset(Dataset):
    def __init__(self, data, tokenizer, max_context_sentences=2, context_type: Literal["prefix", "suffix"] = "prefix"):
        self.data = data
        self.tokenizer = tokenizer
        self.max_context_sentences = max_context_sentences
        self.context_type = context_type

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):
        a, b = self.get_max_context_sample(
            self.data["input_ids"],
            sentence_idx=idx,
            max_length=self.tokenizer.max_len_sentences_pair,
            max_context_sentences=self.max_context_sentences,
            context_type=self.context_type,
        )

        encoding = self.tokenizer.prepare_for_model(a, b, return_tensors="pt", add_special_tokens=True, padding=False)

        encoding["label"] = self.data["label"][idx]

        return encoding


    @staticmethod
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
            The maximum context sample and the offset.
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


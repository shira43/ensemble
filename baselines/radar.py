# Modified from: RAID, Dugan et al. 2024
# > https://github.com/liamdugan/raid/blob/main/detectors/models/radar/radar.py

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from utils import DetectorABC, run_detector

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, ERROR
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class Radar(DetectorABC):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(
            AutoTokenizer.from_pretrained("./radar-vicuna-7b"),
            device=device,
        )
        self.device = torch.device(device)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "./radar-vicuna-7b",
        )
        self.model.eval()
        self.model.to(self.device)

    def tokenize(self, texts: list[str]) -> BatchEncoding:
        return self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=512,
            return_length=True,
        )

    @torch.inference_mode()
    def predict(self, inputs: dict) -> list[float]:
        encoding = self.tokenizer.pad(inputs, return_tensors="pt").to(self.device)
        outputs = self.model(**encoding)
        output_probs = F.log_softmax(outputs.logits, -1)[:, 0].exp().tolist()
        return output_probs

    def process(self, inputs: dict) -> dict[str, list[float]]:
        return {
            "prediction": self.predict(
                {
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                }
            )
        }

    @torch.inference_mode()
    def inference(self, texts: list) -> list:
        predictions = []
        for text in tqdm(texts):
            with torch.no_grad():
                inputs = self.tokenizer(
                    [text],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                output_probs = (
                    F.log_softmax(self.model(**inputs).logits, -1)[:, 0].exp().tolist()
                )
            predictions.append(output_probs[0])
        return predictions


if __name__ == "__main__":
    import os
    from pathlib import Path
    import gc
    from datasets import load_dataset, disable_caching


    logging.info("Program started.")

    logging.info("Setting jsonl file path.")
    jsonl_path = Path(__file__).resolve().parent.parent / "testSeq.jsonl"

    logging.info("Loading dataset.")
    data = load_dataset("json", data_files=str(jsonl_path), split="train")
    logging.info("Dataset loaded as Huggingface Dataset.")


    def run_radar():
        results = run_detector(Radar(device="cuda"), data)
        return results

    logging.info("Running radar...")
    scores_radar = run_radar()
    logging.info(f"Running radar is completed.")
    logging.info(f"Radar score: {scores_radar}")
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    scores_radar
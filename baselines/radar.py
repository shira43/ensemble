# Modified from: RAID, Dugan et al. 2024
# > https://github.com/liamdugan/raid/blob/main/detectors/models/radar/radar.py

import torch
import json
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from utils import DetectorABC, run_detector
from datasets import Dataset, DatasetDict, disable_caching

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

    from pathlib import Path
    from datasets import disable_caching, load_dataset
    import gc

    this_dir = Path(__file__).resolve().parent
    root_dir = this_dir.parent
    jsonl_path = root_dir / "testSeq.jsonl"

    with open(jsonl_path, "r") as f:
        data_list = [json.loads(line) for line in f]

    data = Dataset.from_list(data_list)


    def run_radar():
        results = run_detector(Radar(device="cuda"), data)
        return results


    scores_radar = run_radar()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    scores_radar
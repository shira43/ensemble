import argparse
import json
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from seqXGPT.backend_utils import BBPETokenizerPPLCalc


class LocalBackendSniffer:
    """
    Uses the same BBPETokenizerPPLCalc as the original inference server,
    but runs everything locally – no HTTP, no mosec.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda:1" if torch.cuda.is_available() else "cpu"):
        # load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

        # build the *exact* ppl calculator the authors used
        byte_encoder = bytes_to_unicode()
        self.ppl_calc = BBPETokenizerPPLCalc(
            byte_encoder, self.model, self.tokenizer, self.device
        )

    @torch.no_grad()
    def get_features(self, text: str):
        """
        Returns the triple expected by gen_features.py:
            [loss, begin_word_idx, ll_tokens]
        """
        return self.ppl_calc.forward_calc_ppl(text)


def gen_features(input_file, output_file):
    """replicates get_features minimally from gen_features.py (SeqXGPT) so it runs locally
    input_file: str -> name of input file path with the text data (format of jsonl {text, prompt_len, label})
    output_file: str -> name of output file"""

    # instead of call to inference server
    sniffer = LocalBackendSniffer()

    en_labels = {
        'api': 1,
        'user_and_api': 2,
        'human': 0
    }

    # line example: {"text": "Hello World.", "prompt_len": 0, "label": "api"}
    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    print('input file: {}, length: {}'.format(input_file, len(lines)))
    print("The features for the SeqXGPT Model are being generated. This may take a while...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']
            label = data['label']
            prompt_len = data['prompt_len']
            losses = []
            begin_idx_list = []
            ll_tokens_list = []

            if label not in en_labels:
                print(f"Skipping unknown label: {label}")
                continue
            label_int = en_labels[label]

            loss, begin_word_idx, ll_tokens = sniffer.get_features(line)

            losses.append(loss)
            begin_idx_list.append(begin_word_idx)
            ll_tokens_list.append(ll_tokens)

            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'label_int': label_int,
                'label': label,
                'text': line,
                'prompt_len': prompt_len
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')



# Function to append a document to the formatted data list
def append_document(formatted_data, text, prompt_len, label):
    if text:
        print('='*100)
        print('\n\n\n')
        print(text[:prompt_len])
        print('\n\n\n')
        print(text[prompt_len:])
        print(label)
        print('\n\n\n')
        formatted_data.append({
            "text": text,
            "prompt_len": prompt_len,
            "label": label
        })

def dataset_split_to_pandas(hf_dataset):
    """
    Splits huggingface dataset into train and test set and converts to pandas dataframe.
    we also filter out all columns where label = -1 (prompt)

    :param hf_dataset: huggingface dataset
    :return: train, validation, test pandas dataframe
    """

    train_1 = hf_dataset["train"].to_pandas()
    train_2 = hf_dataset["validation"].to_pandas()
    train_set = pd.concat([train_1, train_2], ignore_index=True)
    train_set = train_set[train_set["label"] != -1]

    test_set = hf_dataset["test"].to_pandas()
    test_set = test_set[test_set["label"] != -1]

    return train_set, test_set


def obtain_jsonl(data_df, output_path):
    """
    Function from SeqXGPT process_data.py

    :param data_df: contains at least columns: {session_id: unique ID for each dialogue session,
                                                sentence_text: a single utterance,
                                                sentence_source: "user", "api", "system" etc.}
    :param output_path: file path of output .jsonl
    :return: None, just writes jsonl to output_path
    """

    # Process the data to create multiple documents per session with varying labels
    formatted_data = []
    current_session = ""
    current_text = ""
    prompt_len = 0
    current_label = ""

    for idx, entry in data_df.iterrows():
        if entry["session_id"] != current_session:
            # Append the previous document if it exists
            if current_label == 'user':
                current_label = 'api'
            append_document(formatted_data, current_text, prompt_len, current_label)

            # Start a new document
            current_session = entry["session_id"]
            current_text = entry["sentence_text"]
            current_label = entry["sentence_source"]
            prompt_len = len(current_text) if current_label == "user" else 0
        else:
            if current_label != entry["sentence_source"]:
                if current_label == "user":
                    prompt_len = len(current_text)
                    current_text += " " + entry["sentence_text"]
                else:

                    if entry["sentence_source"] == "user":
                        # non-user to user
                        append_document(formatted_data, current_text, prompt_len, current_label)
                        current_text = entry["sentence_text"]
                        prompt_len = len(current_text)
                        current_label = entry["sentence_source"]
                    else:
                        # non-user to another non-user
                        append_document(formatted_data, current_text, prompt_len, current_label)
                        current_text = entry["sentence_text"]
                        prompt_len = 0
                        current_label = entry["sentence_source"]

                current_label = entry["sentence_source"]
            else:
                if entry["sentence_source"] == "user":
                    current_text += " " + entry["sentence_text"]
                    prompt_len = len(current_text)
                else:
                    current_text += " " + entry["sentence_text"]

    if current_label == 'user':
        current_label = 'api'
    # Append the last document
    append_document(formatted_data, current_text, prompt_len, current_label)

    # Convert the result to JSON format
    jsonl_data = "\n".join(json.dumps(doc) for doc in formatted_data)

    # write to jsonl file
    with open(output_path, 'w') as f:
        f.write(jsonl_data)
        f.close()


def label_to_source(example):
    label_map = {
        0: "user",
        1: "api",
        2: "user_and_api",
    }
    return {"sentence_source": label_map.get(example["label"], "unknown")}


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_file", type=str, help="input file", defaul=None)
    # parser.add_argument("--output_folder", type=str, help="output file", default=None)
    parser.add_argument("--dataset", type=str, default='coauthor-zeng', choices=["pasted-base", "coauthor-base", "coauthor-extended", "coauthor-zeng"])
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()

    dataset = args.dataset

    if dataset == "coauthor-zeng":
        coauthor_zeng = load_dataset("43shira43/coauthor-zeng")
        train, test = dataset_split_to_pandas(coauthor_zeng)
        obtain_jsonl(train, "seqXGPT/dataset/coauthor_zeng/train.jsonl")
        obtain_jsonl(test, "seqXGPT/dataset/coauthor_zeng/test.jsonl")
        gen_features("seqXGPT/dataset/coauthor_zeng/train.jsonl", "seqXGPT/dataset/coauthor_zeng/train_features.jsonl")
        gen_features("seqXGPT/dataset/coauthor_zeng/test.jsonl", "seqXGPT/dataset/coauthor_zeng/test_features.jsonl")
    elif dataset == "pasted-base":
        pasted = load_dataset("43shira43/pasted-base")
        pasted = pasted.map(label_to_source)
        pasted.rename_columns({
            "id": "session_id",
            "text": "sentence_text"
        })
        train, test = dataset_split_to_pandas(pasted)
        obtain_jsonl(train, "seqXGPT/dataset/pasted_base/train.jsonl")
        obtain_jsonl(test, "seqXGPT/dataset/pasted_base/test.jsonl")
        gen_features("seqXGPT/dataset/pasted_base/train.jsonl", "seqXGPT/dataset/pasted_base/train_features.jsonl")
        gen_features("seqXGPT/dataset/pasted_base/test.jsonl", "seqXGPT/dataset/pasted_base/test_features.jsonl")
    elif dataset == "coauthor-base": #session id zu id, sentence_text und sentence_source (label_str)
        coauthor_base = load_dataset("43shira43/coauthor-base")
        train, test = dataset_split_to_pandas(coauthor_base)
        obtain_jsonl(train, "seqXGPT/dataset/coauthor_base/train.jsonl")
        obtain_jsonl(test, "seqXGPT/dataset/coauthor_base/test.jsonl")
        gen_features("seqXGPT/dataset/pasted_base/train.jsonl", "seqXGPT/dataset/pasted_base/train_features.jsonl")
        gen_features("seqXGPT/dataset/pasted_base/test.jsonl", "seqXGPT/dataset/pasted_base/test_features.jsonl")
    elif dataset == "coauthor-extended":
        coauthor_extended = load_dataset("43shira43/coauthor-extended-base")
        coauthor_extended.rename_columns({
            "id": "session_id",
            "text": "sentence_text",
            "label_str": "sentence_source"
        })
        train, test = dataset_split_to_pandas(coauthor_extended)
        obtain_jsonl(train, "seqXGPT/dataset/coauthor_extended/train.jsonl")
        obtain_jsonl(test, "seqXGPT/dataset/coauthor_extended/test.jsonl")
        gen_features("seqXGPT/dataset/coauthor_extended/train.jsonl", "seqXGPT/dataset/coauthor_extended/train_features.jsonl")
        gen_features("seqXGPT/dataset/coauthor_extended/test.jsonl", "seqXGPT/dataset/coauthor_extended/test_features.jsonl")
    else:
        print("DATASET NOT FOUND")
        ValueError("Please specify a valid dataset: coauthor-zeng, pasted-base, coauthor-base or coauthor-extended")

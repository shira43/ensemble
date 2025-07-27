import argparse
import sys
from argparse import Namespace

import pandas as pd
#from fairseq.benchmark.benchmark_multihead_attention import BATCH
from tqdm import tqdm

from datasets import Dataset, load_dataset
from seqXGPT.SeqXGPT.dataloader import DataManager
from seqXGPT.SeqXGPT.model import ModelWiseTransformerClassifier
from wrapper_helper_seqxgpt import gen_features
from seqXGPT.dataset.process_data import obtain_jsonl
#sys.path.append("seqXGPT/SeqXGPT")
#from seqXGPT.SeqXGPT.train import SupervisedTrainer

import torch

# bmes labels for en_labels (api, user_api and human)
id2label = {0: 'B-api', 1: 'M-api', 2: 'E-api', 3: 'S-api', 4: 'B-user_and_api', 5: 'M-user_and_api', 6: 'E-user_and_api', 7: 'S-user_and_api', 8: 'B-human', 9: 'M-human', 10: 'E-human', 11: 'S-human'}

en_labels = {
    'api': 0,
    'user_and_api': 1,
    'human': 2
}

class SeqXGPTWrapper:
    def __init__(self, input_file, output_file, train_path, test_path, ckpt_name="seqXGPT/linear_en.pt", batch_size=32,
                 id2label=id2label, seq_len=1024, num_train_epochs=20, weight_decay=0.1 , lr=5e-5 , warm_up_ratio=0.1):
        self.input_file = input_file
        self.output_file = output_file
        self.ckpt_name = ckpt_name
        self.seq_len = seq_len

        # Create a Namespace object mimicking argparse
        args = Namespace(
            seq_len=seq_len,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            lr=lr,
            warm_up_ratio=warm_up_ratio
        )

        self.data = DataManager(train_path=train_path, test_path=test_path, batch_size=batch_size, max_len=seq_len, human_label='human', id2label=id2label)
        # can be changed to other models but needs to be imported then from seqXGPT.SeqXGPT.model
        self.model = ModelWiseTransformerClassifier(id2labels=id2label, seq_len=seq_len)
        self.trainer = SupervisedTrainer(self.data, self.model, en_labels, id2label, args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)


    def load_detector(self, ckpt_name="seqXGPT/linear_en.pt"):
        """
        Loads a full Model from Checkpoint.

        :param ckpt_name: name of path to previously saved checkpoint.
        :return: model (ready for inference and resuming of training)
        """

        loaded_model = torch.load(ckpt_name, map_location="cpu")
        loaded_model = loaded_model.to(self.device).eval()
        self.model = loaded_model

        return self.model


    def _compress_logits(self, logits, labels):
        """
        Compresses the logits from (batch_size, seq_len, num_classes=12) to (batch_size, num_classes=3), to follow more standard conventions

        :param logits: logits tensor of seqXGPT forward pass (batch_size, seq_len, num_classes=12) -> classes: id2label
        :param labels: labels of features (needed to find end of tokens and beginning of pad tokens)
        :return: logits tensor (batch_size, num_classes=3)  -> classes (api, user_and_api, human)
        """

        batch_size = logits.size(0)
        final_logits = torch.zeros(batch_size, 3)

        for batch_idx, sequence in enumerate(logits):  # sequence: [seq_len, num_classes]
            seq_labels = labels[batch_idx].tolist()
            try:
                first_pad_idx = seq_labels.index(-1)
                temp_logits = torch.zeros(first_pad_idx, 3)
            except ValueError:
                first_pad_idx = self.seq_len + 1
                temp_logits = torch.zeros(self.seq_len, 3)

            for token_idx, token_logits in enumerate(sequence):  # token_logits: [num_classes]
                if token_idx >= first_pad_idx:
                    break
                else:
                    if token_idx == 0:
                        # if it's the first token we take the logit of class B-
                        new_logits = token_logits[[0,4,8]]
                        temp_logits[token_idx] = new_logits
                    else:
                        # otherwise we take logit of class M-
                        new_logits = token_logits[[1,5,9]]
                        temp_logits[token_idx] = new_logits

            # aggregate over each column to get mean logit
            api_logit = temp_logits[:, 0].sum() / first_pad_idx
            user_and_api = temp_logits[:, 1].sum() / first_pad_idx
            human = temp_logits[:, 2].sum() / first_pad_idx

            aggregate_logits = torch.tensor([[api_logit, user_and_api, human]])

            final_logits[batch_idx] = aggregate_logits

        return final_logits


    def predict_logits(self, input_data):
        """

        :param input_data: pandas dataframe with text, label and prompt_len
        :return: torch tensor -> logits of model.  -> currently size (batch_size * seq_len * num_labels)
        """

        # get model ready for inference
        self.load_detector(self.ckpt_name)

        total_logits = []

        input_data_path = "datasets/base/seqXGPT/temp_formatted_data.jsonl"
        output_data_path = "datasets/base/seqXGPT/returned_features.jsonl"
        # turn pandas df to jsonl so we have expected format for gen_features
        input_data.to_json(input_data_path, orient='records', lines=True)
        gen_features(input_data_path, output_data_path)

        # initialize dataset
        # .jsonl file → initialize_dataset() → samples_dict → Dataset.from_dict() → DataLoader
        samples_dict = self.data.initialize_dataset(output_data_path)
        dataset = Dataset.from_dict(samples_dict)
        dataloader = self.data.get_eval_dataloader(dataset)

        # inputs have format from DataManager data_collator {features, labels, text}
        for step, inputs in enumerate(tqdm(dataloader, desc="Iteration")):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.no_grad():
                output = self.model(inputs['features'], inputs['labels'])
                logits = output['logits']

                smaller_logits = self._compress_logits(logits, inputs['labels'])
                total_logits.append(smaller_logits.cpu())

        return torch.cat(total_logits, dim=0)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coauthor-zeng', choices=['coauthor-zeng', 'coauthor-extended-np'])
    parser.add_argument('--only_gen_features', type=bool, default=True)


    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    if args.only_gen_features:
        dataset_name = f"43shira43/{args.dataset}"
        data_df = load_dataset(dataset_name)
        # Filter out possible prompts
        data_df = data_df.filter(lambda example: example["label"] in [0, 1, 2])

        for split in data_df.keys():
            if "text" in data_df[split].column_names:
                data_df[split] = data_df[split].rename_columns({
                    "text": "sentence_text",
                    "id": "session_id",
                    "source": "sentence_source", })

        train_df = data_df["train"].to_pandas()
        val_df = data_df["validation"].to_pandas()
        final_train_df = pd.concat([train_df, val_df], ignore_index=True)

        test_df = data_df["test"].to_pandas()

        if args.dataset == 'coauthor-zeng':
            #generate train features
            train_jsonl = obtain_jsonl(final_train_df, 'datasets/seqXGPT/coauthor/train')
            train_output = "datasets/seqXGPT/coauthor/train_features.jsonl"
            gen_features(train_jsonl, train_output)

            #generate test features
            test_jsonl = obtain_jsonl(test_df, 'coauthor/coauthor/test')
            test_output = "datasets/seqXGPT/coauthor/test_features.jsonl"
            gen_features(test_jsonl, test_output)




        else:
            train_jsonl = obtain_jsonl(final_train_df, 'datasets/seqXGPT/coauthor-extended/train')
            train_output = "datasets/seqXGPT/coauthor-extended/train_features.jsonl"
            gen_features(train_jsonl, train_output)

            test_jsonl = obtain_jsonl(train_df, 'datasets/seqXGPT/coauthor-extended/test')
            test_output = "datasets/seqXGPT/coauthor-extended/test_features.jsonl"
            gen_features(test_jsonl, test_output)

            print("Finished generating features. Look in datasets folder.")

    else:
        wrapper = SeqXGPTWrapper("input", "out", "seqXGPT/dataset/coauthor/train.jsonl", "seqXGPT/dataset/coauthor/val.jsonl")


        # TODO add column prompt_len = 0 to all data of df
        # TODO change to correct .jsonl so test works.
        x_data = pd.read_json("testSeq.jsonl", orient='records', lines=True)
        logits = wrapper.predict_logits(x_data)
        print(logits)
        print(logits.shape)


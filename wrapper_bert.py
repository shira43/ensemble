import pandas as pd
import torch
from bert.model.models import BertClassifier, DebertaClassifier
from wrapper_helper_bert import build_graph, prepare_for_graph

class BertWrapper:
    def __init__(self, ckpt_name: str, bert_init: str, nb_class: int = 20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_name = ckpt_name
        self.bert_init = bert_init
        self.nb_class = nb_class
        self.tokenizer = None

        if "deberta" in bert_init.lower():
            self.model = DebertaClassifier(bert_init, nb_class)
        else:
            self.model = BertClassifier(bert_init, nb_class)


    def load_detector(self, ckpt_name: str = None, nb_class: int = 20):
        """
        Loads a Model from Checkpoint.

        :param ckpt_name: name of path to previously saved checkpoint.
        :return: model (ready for inference and resuming of training)
        """
        if "deberta" in self.bert_init.lower():
            self.model = DebertaClassifier(self.bert_init, nb_class)
        else:
            self.model = BertClassifier(self.bert_init, nb_class)


        if ckpt_name is None:
            ckpt_name = self.ckpt_name

        ckpt = torch.load(ckpt_name, map_location=self.device)

        self.model.bert_model.load_state_dict(ckpt['bert_model'])

        if isinstance(self.model, BertClassifier):
            self.model.classifier.load_state_dict(ckpt['classifier'])

        self.model.to(self.device).eval()

        # Use the tokenizer that lives inside the classifier instance
        self.tokenizer = self.model.tokenizer

        return self.model

    @torch.no_grad()
    def predict_logits(self, input_data, batch_size: int = 64, max_length: int = 128):
            """
            texts : iterable[str] | pd.Series
            returns: torch.FloatTensor (len(texts), num_labels)
            """

            # Preprocess data
            prepare_for_graph(input_data)
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = build_graph()

            nb_node = adj.shape[0]
            nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
            nb_word = nb_node - nb_train - nb_val - nb_test
            nb_class = y_train.shape[1]

            # load detector with correct nb_class (calculated from data)
            self.load_detector(self.ckpt_name, nb_class)

            texts = input_data["text"]
            all_logits = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)

                logits = self.model(input_ids, attention_mask)  # (B, nb_class)
                all_logits.append(logits.cpu())

            return torch.cat(all_logits, dim=0)



if __name__ == "__main__":
    # TODO checkpoint für bert erstellen
    wrapper = BertWrapper("ckpt/bert.ckpt", "bert-base-uncased")
    x_data = pd.read_json("testSeq.jsonl", orient='records', lines=True)
    logits = wrapper.predict_logits(x_data)
    print(logits)
    print(logits.shape)


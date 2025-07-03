import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
from seqXGPT.backend_utils import BBPETokenizerPPLCalc


class LocalBackendSniffer:
    """
    Uses the same BBPETokenizerPPLCalc as the original inference server,
    but runs everything locally – no HTTP, no mosec.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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

    en_labels = {
        'api': 0,
        'user_and_api': 1,
        'human': 2
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
            label_int = en_labels[label]

            # instead of call to inference server
            sniffer = LocalBackendSniffer()
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


# if __name__ == "__main__":
#
#     gen_features("testSeq.jsonl", "outputTestSeq.jsonl")

# Instructions for Running the Models

This project compares several detection methods for identifying human, AI, and co-written texts.  

The primary datasets used are the **CoAuthor** and **CoAuthor-Extended** datasets, available on [HuggingFace](https://huggingface.co/43shira43), along with additional datasets.  

Before running any models, make sure to install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## SeqXGPT
**Original repository:** [SeqXGPT](https://github.com/Jihuai-wpy/SeqXGPT)

### Data Preparation
This repository includes two additional files not present in the original:
- `wrapper_seqxgpt.py`
- `wrapper_helper_seqxgpt.py`

These files simplify feature generation for SeqXGPT.  

- `wrapper_seqxgpt.py` first saves the dataset in the correct format (see `seqXGPT/README.md`), and then generates features directly, without needing to download the inference server.  
- Datasets are saved in `datasets/seqXGPT/{coauthor | coauthor-extended}`, and automatically split into train and test sets.  
- To use other datasets, modify `wrapper_seqxgpt.py` accordingly.  
- It is also possible to extract logits using `wrapper_seqxgpt.py` (see the comments at the end of the file for an example).

### Running SeqXGPT
Once the data is correctly formatted and features are generated, train SeqXGPT with:

```bash
python SeqXGPT/train.py \
    --gpu 0 \
    --train_path dataset/coauthor/train.jsonl \
    --test_path dataset/coauthor/test.jsonl \
    --batch_size 32 \
    --num_train_epochs 12
```

For more details, refer to `seqXGPT/README.md`.

---

## Context BERT
The `context_bert` folder contains the extended models presented in this study:  
- **BERT-token**  
- **BERT-context**

### Running Context BERT
Models can be launched with the desired arguments.  
The full list of arguments is defined in `context_bert/helpers.py` (`argparse()` function).  

Example call:

```bash
python finetune_token_bert.py --dataset coauthor-extended-np --batch_size 16 --nb_epochs 5
```

---

## BERT Models
**Original repository:** [AISentenceDetection](https://github.com/douglashiwo/AISentenceDetection)  
**Corresponding study:** *"Detecting AI-Generated Sentences in Human-AI Collaborative Hybrid Texts: Challenges, Strategies, and Insights"* by Zeng et al.

### Data Preparation
The BERT models require preprocessing similar to the original study:

Inside `bert/data/coauthor`:
```bash
python coauthor_to_train_data.py --dataset coauthor-extended-np
```

This converts the HuggingFace dataset into the required format.  

Inside `bert`:
```bash
python build_graph.py we
```

This builds the text graph dataset.  
If you switch datasets (e.g., to `coauthor-zeng`), rerun the above steps with the new dataset name.

For details, see `bert/README.md`.

### Running BERT Models
- Run with original fine-tuning (as in Zeng et al.):  
  ```bash
  python finetune_bert_original.py
  ```
- Run with **weighted random sampling**:  
  ```bash
  python finetune_bert_weighted.py [arguments]
  ```

---

## Baselines
The `baselines` folder contains implementations for **Binoculars** and **Radar**.

### Binoculars
**Original repository:** [RAID - Binoculars](https://github.com/liamdugan/raid/blob/main/detectors/models/binoculars/utils/metrics.py)  

This implementation was adapted by *Manuel Schaaf* to support **multiclass evaluation**.  
The only parameter required is the dataset.  

Example:
```bash
python binoculars.py --dataset coauthor-extended-np
```

---

### Radar
**Original repository:** [RAID - Radar](https://github.com/liamdugan/raid/blob/main/detectors/models/radar/radar.py)  
Originally binary only. For multiclass evaluation, this adaptation from **MixSet** is used:  
[MixSet Radar](https://github.com/Dongping-Chen/MixSet/blob/main/methods/radar.py)

Run Radar in the same way as Binoculars (specifying only the dataset):  

Example:
```bash
python radar_mixset.py --dataset coauthor-extended-np
```

---

## Summary
- **SeqXGPT** → Preprocess with wrappers, then train with `train.py`.  
- **Context BERT** → Directly run with desired arguments.  
- **BERT Models** → Preprocess datasets, then run `finetune_bert_original.py` or `finetune_bert_weighted.py`.  
- **Baselines (Binoculars & Radar)** → Run with dataset argument for multiclass evaluation.  

---
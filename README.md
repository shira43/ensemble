# Instructions for running the models


This project was done to compare several detection methods on how well they can detect human,AI and co-written texts.

This project uses mainly the Coauthor and CoAuthor-Extended Dataset. Which can be found on Huggingface.
(https://huggingface.co/43shira43), as well as additional datasets. 

First run the requirements.txt

Below are instructions for starting the several models:


### SeqXGPT
The original repository is this: https://github.com/Jihuai-wpy/SeqXGPT

##### Data Preparation
The files <br />
wrapper_seqxgpt.py <br />
wrapper_helper_sexgpt.py <br />
were not in the original SeqXGPT repository. They were added, so the generating of the 
features for SeqXGPT will be easier. wrapper_seqxgpt.py uses the helper to generate the features
It first saves the data first in the correct format for SeqXGPT (for further information read the README.md within the 
seqXGPT folder), and then generates the features, without needing to download the inference server first. 
The dataset saved into the correct format can be found inside the folder
datasets/seqXGPT and then either coauthor or coauthor-extended.
The dataset will be automatically splitted into train and test dataset.
If different datasets should be implemented, then the code of wrapper_seqxgpt.py needs
to be adapted accordingly. 
It is also possible to obtain the logits of SeqXGPT through wrapper_seqxgpt.py (see last comment in file for example).

##### Running SeqXGPT
If the data is in the correct format, and the features where generated as stated above,
SeqXGPT can be trained in the following way:


python SeqXGPT/train.py\
    --gpu 0 \
    --train_path dataset/coauthor/train.jsonl\
    --test_path dataset/coauthor/test.jsonl\
    --batch_size 32 \
    --num_train_epochs 12

For more information see seqXGPT/README.md.


### Context BERT
The models inside the context_bert folder where the models, presented in the Study as the expansions. 
BERT-token and BERT-context.

the models can simply be run by calling them with the wanted arguments.
The list of arguments that can be passed is found in the context_bert/helpers.py script in the 
argsparse() function. 

Examplary call:
python finetune_token_bert.py --dataset coauthor-extended-np --batch_size 16 --nb_epochs 5



### BERT models
This repository is from: https://github.com/douglashiwo/AISentenceDetection
The corresponding study was:
"Detecting AI-Generated Sentences in Human-AI Collaborative Hybrid Texts: Challenges, Strategies, and Insights" from Zeng et. al

##### Data preparation
First the data needs to be prepared in a certain way, in order to run the BERT models presented in the Study of Zeng et al.

Inside of bert/data/coauthor:
1. Run `python coauthor_to_train_data.py --dataset coauthor-extended-np` to transform the huggingface dataset into the required format.

Inside of bert:
2. Run `python build_graph.py we` to build the text dataset.

if the dataset should be changed to coauthor-zeng for example, then this needs to be rerun with the correct dataset name.

For further information read the bert/README.md

###### Running BERT models
To run the models with the original finetuning of BERT, how it was done in the study of 

run python finetune_bert_original.py if the model should behave like the ones in the original study.
(See bert/README.md for further details)

run python finetune_bert_weighted.py with the neccessary arguments if the model should be run with the weighted random sampler. 

### Baselines
The baselines folder holds the Binoculars and Radar code. 

##### Binoculars
Original repository for Binocualrs: https://github.com/liamdugan/raid/blob/main/detectors/models/radar/radar.py

The Binoculars implementation how it is in this repository now was adapted by Manuel Schaaf.
Adapted for this code was only that binoculars can also evaluate multiclass now.

For Binoculars there is only the parameter of the dataset that can be given.
So it is run for example by: python binoculars.py --dataset coauthor-extende-np

##### Radar
Original repository for Radar: https://github.com/liamdugan/raid/blob/main/detectors/models/radar/radar.py
But originally it was only binary, the authors from MixSet already adapted it for multiclass,
therefore this adaptation was used:
https://github.com/Dongping-Chen/MixSet/blob/main/methods/radar.py

It is run in the same way like Binoculars (only the dataset needs to be specified):

So for the CoAuthor-Extended dataset it would be: python radar_mixset.py --dataset coauthor-extended-np


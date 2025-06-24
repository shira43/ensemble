# BERT style models for the CoAuthor dataset

## Introduction
This code repository is used to train and test the performance of BERT-style models on the coAuthor dataset.

## Models

BERT style model:`bert-base-uncased`,`roberta-base`,`distilbert-base-uncased`,`deberta-v3-base`,`deberta-base` .


## Usage

### 1. Create a new anaconda environment and install the dependencies:

1. In a command line terminal, create a new conda environment using the following command:

    ```bash
    conda env create -f environment.yml
    ```
    This will create a new conda environment using the information from the `environment.yml` file.

2. Or create and activate the new environment:

   ```bash
   conda create -n BERT python=3.9

   source activate BERT
    ```

    Then, install pip dependencies from the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```


### 2. Prepare the dataset

1. Run `python coathor_to_train_data.py --dataset we` to transform `20231114_coauthor_data.xlsx` dataset into text data`./data/corpus/we.clean.txt` and data labels file`./data/we.txt`

2. Run `python build_graph.py we` to build the text dataset.

### 3. Train and test models

1. Modify the code to suit run environment.

   * Set the local BERT path in `finetune_bert.py` or modify it to huggingface model name.
   ```python
   def get_initi_local_bert_path(bert_init):
       # Create a dictionary that maps the model name to the path of the pre-trained model
       pretrained_model_paths = {
           "roberta-base": "./pre_trained_model/roberta_base",
           "bert-base-uncased": "./pre_trained_model/bert-base-uncased",
           "distilbert-base-uncased": "./pre_trained_model/distilbert-base-uncased",
           "deberta-v3-base":"./pre_trained_model/deberta-v3-base",
           "deberta-base":"./pre_trained_model/deberta-base",
       }
   ```

   * Then, set local data xlsx path in `finetune_bert.py`.

   ```python
   def save_test_dataset_data(prediction_data,merged_data_path,dataset):
       if dataset == "we":
           data_table_path = 'data/coauthor/20231114_coauthor_data_real.xlsx'
       elif dataset == "we_test":
           data_table_path = 'data/coauthor/20231114_coauthor_data_test.xlsx'
   ```


2. If you need to do training and testing tasks through bash scripts, please run from sh scrips from  `./run_scripts/run_examples.sh`. such as follows.
   
   For example, the purpose of the bash script is to train roberta-base on the `we` (coauthor) dataset. 
   The model is set to randomly sample 20,000 samples from the training data in each epoch. 
   The learning rate of the model is set to {0.0001 0.00002 0.00005} 
   and each learning rate is run 5 times.
   
   In detail, 
   ```bash
   #!/bin/bash
   
   cd /project/
   # Train and test BERT
   dataset="we"
   epoch_sample_num=20000
   bert_init="bert-base-uncased"
   checkpoint_dir="./checkpoint_test/${bert_init}/${dataset}"
   is_save_model="no_save"
   nb_epochs=10
   # Define different learning rate
   for bert_lr in 0.0001 0.00002 0.00005; do
     for batch_size in 32; do
         for time in 1 2 3 4 5; do
   #        Python interpreter file path of the virtual environment
             command="/root/anaconda3/envs/BERT/bin/python3.9 finetune_bert.py"
             command="$command --dataset ${dataset}"
             command="$command --epoch_sample_num ${epoch_sample_num}"
             command="$command --bert_init ${bert_init}"
             command="$command --checkpoint_dir ${checkpoint_dir}_lr_${bert_lr}_time_${time}"
             command="$command --bert_lr ${bert_lr}"
             command="$command --batch_size ${batch_size}"
             command="$command --is_save_model ${is_save_model}"
             command="$command --nb_epochs ${nb_epochs}"
             # Print and execute the command
             echo "Executing command: $command"
             eval $command
         done
     done
   done 
   ```


3. If you want to run a single model, please run `python finetune_bert.py --dataset we --epoch_sample_num 20000 --bert_init bert-base-uncased --checkpoint_dir ./checkpoint_test/bert-base-uncased/we_lr_0.00001_time_no --bert_lr 0.00001 --batch_size 32` 
to finetune the BERT model over the target dataset. The model and training logs will be saved to `checkpoint/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.
   
   
4. If you need to run your training and testing code in the background, use the `screen` software. In detail,
   1. Create a screen
   `screen -S ***`
   2. View the list of current screens
   `screen -ls`
   3. Re-enter the screen you have already created
   `screen -r ***`


6. Get the training and test results from the file path `./checkpoint_test/` which saved prediction results and experimental results for all models. Run `get_result_table.py` to integrate all the results.


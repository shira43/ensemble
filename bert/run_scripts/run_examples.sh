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
#        File path of the python interpreter for the virtual environment
          command="/root/anaconda3/envs/BERT-GCN/bin/python3.9 finetune_bert.py"
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


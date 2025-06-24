## results of BERT style model
### Model list
bert-base, roberta_base, distilbert-base-uncased

### File Naming format

* Sh script: ${dataset_name}_lr_${bert_lr}_time_${time}
* example: we_lr_0.0001_time_1

## results of BERT with gcn or gat
### Model list
GCN (Graph convolution Network) or gat

### File Naming format
* Sh script:${BERT_INIT}_${GCN_MODEL}_${DATASET}_balance_${BALANCE}_bert_${bert_lr}_gcn_lr_${gcn_lr}_time_${TIME}
* example: roberta_base_gat_we_balance_0.3_bert_0.00002_gcn_lr_0.001_time_1


## results computing

`python get_result_table.py ` Run the code to aggregate all the data.
 
1. Traverse the directory of all files in the current folder, keep the relative file path of all files named training.log, and store it in the list named result_path_list.
2. Create a pandas table named results_table, iterate over each file path file_path of results_path_list, the first folder name of file_path string is stored in the column named model_name in results_table.
3. The folder name of the second file_path string is stored in the column named model_parameters in the results_table.
4. Read through the file for each file_path, reading from the back of the text file to get the line before the penultimate "New checkpoint" character.
5. The results_text reads as follows: "Epoch: 2 Test dataset: Test_acc: 0.6889". This string is parsed into a dictionary and stored in the rows of the results_table, where each column is the dictionary key and the value is the dictionary value.

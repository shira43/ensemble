import pandas as pd
import nltk
from openpyxl import load_workbook
import argparse, shutil, logging

# Make sure you have downloaded NLTK's word segmentation resource
# nltk.download('punkt')

def reorder_dataframe(df, column_name):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    train_rows = df[df[column_name] == 'train']
    valid_rows = df[df[column_name] == 'valid']
    test_rows = df[df[column_name] == 'test']
    reordered_df = pd.concat([train_rows, valid_rows, test_rows], ignore_index=True)
    return reordered_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # we = writing essay dataset
    parser.add_argument('--dataset', default='we', choices=['20ng', 'R8', 'R52', 'ohsumed', 'mr', "we", "we_test"])
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "we":
        dataset_path = '20231114_coauthor_data.xlsx'
    elif dataset == "we_test":
        dataset_path = '20231114_coauthor_data_test.xlsx'
    data_table = pd.read_excel(dataset_path, engine='openpyxl')
    # Filter out the data and create a copy to avoid SettingWithCopyWarning
    selected_data_table = data_table[
        (data_table['label'].isin([0, 1, 2])) &
        (data_table['train_ix'].isin(['train', 'test','valid']))
    ].copy()
    selected_data_table_sorted = reorder_dataframe(selected_data_table,"train_ix")
    # Use NLTK for text segmentation

    selected_data_table_sorted['split_sentence_text'] = selected_data_table_sorted['sentence_text'].apply(
        lambda x: ' '.join(nltk.word_tokenize(x))
    )

    selected_data_table_sorted['id_index'] = range(len(selected_data_table_sorted))

    with open('../corpus/we.clean.txt'.replace("we", dataset), 'w', encoding='utf-8') as f:
        for sentence in selected_data_table_sorted['split_sentence_text']:
            f.write(sentence + '\n')
    # Write id_index, train_ix, label columns to txt file

    selected_data_table_sorted[['id_index', 'train_ix', 'label']].to_csv(
        '../we.txt'.replace("we", dataset), sep='\t', index=False, header=False
    )
    print(dataset+" Files have been successfully saved.")

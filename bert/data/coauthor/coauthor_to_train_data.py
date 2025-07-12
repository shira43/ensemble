import pandas as pd
import nltk
from openpyxl import load_workbook
import argparse, shutil, logging
from datasets import load_dataset
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


def get_selected_data_table_sorted(ds):
    # apply filter
    selected_data_table = ds.filter(
        lambda example: example["label"] in [0, 1, 2] and example["train_ix"] in ["train", "test", "valid"]
    )

    train_df = selected_data_table["train"].to_pandas()
    val_df = selected_data_table["validation"].to_pandas()
    test_df = selected_data_table["test"].to_pandas()
    selected_data_table_sorted = pd.concat([train_df, val_df, test_df], ignore_index=True)

    text_col = 'sentence_text' if 'sentence_text' in selected_data_table_sorted.columns else 'text'
    # Use NLTK for text segmentation
    selected_data_table_sorted['split_sentence_text'] = selected_data_table_sorted[text_col].apply(
        lambda x: ' '.join(nltk.word_tokenize(x))
    )

    return selected_data_table_sorted

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # we = writing essay dataset
    parser.add_argument('--dataset', default='we', choices=["pasted", "coauthor-base", "coauthor-extended", "we"])
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == "we":
        coauthor_zeng = load_dataset("43shira43/coauthor-zeng")
        sorted_data = get_selected_data_table_sorted(coauthor_zeng)
    elif dataset == "pasted":
        pasted = load_dataset("43shira43/pasted-base")
        sorted_data = get_selected_data_table_sorted(pasted)
    elif dataset == "coauthor-base":
        coauthor_base = load_dataset("43shira43/coauthor-base")
        sorted_data = get_selected_data_table_sorted(coauthor_base)
    elif dataset == "coauthor-extended":
        coauthor_extended = load_dataset("43shira43/coauthor-extended-base")
        sorted_data = get_selected_data_table_sorted(coauthor_extended)
    else:
        print("DATASET NOT FOUND")
        ValueError("Please specify a valid dataset: we, pasted, coauthor-base or coauthor-extended")
        sorted_data = None

    sorted_data['id_index'] = range(len(sorted_data))

    with open('../corpus/we.clean.txt'.replace("we", dataset), 'w', encoding='utf-8') as f:
        for sentence in sorted_data['split_sentence_text']:
            f.write(sentence + '\n')
    # Write id_index, train_ix, label columns to txt file

    sorted_data[['id_index', 'train_ix', 'label']].to_csv(
        '../we.txt'.replace("we", dataset), sep='\t', index=False, header=False
    )
    print(dataset+" Files have been successfully saved.")

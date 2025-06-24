import os
import pandas as pd

# 1. Walk through the directory of all files in the current folder

result_path_list = []
for root, dirs, files in os.walk(''):
    for file in files:
        if file == 'training.log':
            result_path_list.append(os.path.join(root, file))

# 2. Create a pandas table named results_table

results_table = pd.DataFrame(columns=['model_name', 'model_parameters'])

# 3. Iterate over result_path_list and populate results_table

for file_path in result_path_list:
    path_parts = file_path.split(os.sep)
    if len(path_parts) >= 3:
        model_name = path_parts[1]
        model_parameters = path_parts[2]
        results_table = results_table.append({'model_name': model_name, 'model_parameters': model_parameters}, ignore_index=True)

# 4. Read every file_path file

def get_last_results_text(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        reversed_lines = lines[::-1]
        for line in reversed_lines:
            if "New checkpoint" in line:
                # The list.index method returns the index of the first occurrence of the item in the list.
                return reversed_lines[reversed_lines.index(line) + 1].strip()
    return None

# 5. Parse results_text and update results_table

def parse_results_text(results_text):
    results_text = results_text.replace("Test dataset: ","").replace("Test_f1_score :","Test_f1_score:")

    tokens = results_text.split()
    metrics_dict = {}
    # Walk through each key-value pair
    for i in range(0, len(tokens), 2):
        key = tokens[i].replace(':', '')  # Remove the colon
        value = float(tokens[i + 1])  # Converts the value to a floating point number
        metrics_dict[key] = value
    return metrics_dict

# Update results_table

for index, row in results_table.iterrows():
    file_path = result_path_list[index]
    results_text = get_last_results_text(file_path)
    if results_text:
        results_dict = parse_results_text(results_text)
        for key, value in results_dict.items():
            results_table.at[index, key] = value

results_table.to_excel("results_table.xlsx")

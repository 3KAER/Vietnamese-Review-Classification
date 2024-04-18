import csv
from .data_prepropressing import preprocessing

def prepare_data(file_paths):
    print("Preparing data...")
    x_data = []
    y_data = []
    for file_path in file_paths:
        print(f"Processing data from file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                word_list = preprocessing(row[0])
                label = int(row[1])
                if len(word_list) > 0:
                    x_data.append(word_list)
                    y_data.append(label)
    return x_data, y_data

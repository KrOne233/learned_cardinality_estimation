import csv
import os
import pickle
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_sql_csv(sql_csv_file):
    tables = []
    # Load queries
    with open(sql_csv_file, 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
    print("Loaded queries")
    return tables

def get_all_table_names(tables):
    table_names = set()
    for query in tables:
        for table in query:
            table_names.add(table)
    return table_names


def select_sample(table_names, dict_table_filepath, num_samples):
    df_samples = {}
    for t in table_names:
        df = pd.read_csv(dict_table_filepath[t.split(' ')[0]], sep=',', escapechar='\\', encoding='utf-8',
                         low_memory=False, quotechar='"').sample(num_samples)
        df.columns = [c.lower() for c in df.columns]
        df[f"bit_id_{t.split(' ')[1]}"] = [i for i in range(len(df))]
        df_samples[t.split(' ')[1]] = df

    return df_samples



def query_on_sample(df_samples, sql_csv_line):
    predicates = sql_csv_line[2].split(',')
    i = 0
    df_sample = {}
    while i < len(predicates):
        t = predicates[i].split('.')
        t = t[0]
        df_sample[t] = df_samples[t]
        i = i + 3
    i = 0
    while i < len(predicates):
        t = predicates[i].split('.')[0]
        if len(df_sample[t]) == 0:
            i = i+3
            continue
        if predicates[i + 1] == '=':
            predicates[i + 1] = '=='
        df_sample[t] = df_sample[t][
            eval('df_sample[t][predicates[i].split(".")[1]]' + predicates[i + 1] + predicates[i + 2])]
        i = i + 3

    return df_sample[t]


def generate_bitmap(sql_csv_file, dict_table_filepath, num_samples):
    all_tables = load_sql_csv(sql_csv_file)
    table_names = get_all_table_names(all_tables)
    df_samples = select_sample(table_names, dict_table_filepath, num_samples)
    bitmaps = []
    with open(sql_csv_file) as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for line in tqdm(lines):
            tables = line[0].split(',')
            bitmap = np.zeros([len(tables), num_samples], dtype = np.int8)
            df_result = query_on_sample(df_samples, line)
            if len(df_result)==0:
                bitmaps.append(bitmap)
            else:
                i = 0
                for t in tables:
                    index = df_result[f"bit_id_{t.split(' ')[1]}"]
                    bitmap[i][index] = 1
                bitmaps.append(bitmap)
    return bitmaps


sql_csv_file = 'data/gpu_game_fps/fps_sql_train.csv'
dict_table_filepath = {'fps': 'data/gpu_game_fps/fps_num_lower.csv'}
num_samples = 1000
bitmaps = generate_bitmap(sql_csv_file, dict_table_filepath, num_samples)
with open('data/gpu_game_fps/fps_sql_train' + '.samplebitmap', 'wb') as f:
    pickle.dump(bitmaps, f)



sql_csv_file = 'data/gpu_game_fps/fps_sql_test.csv'
dict_table_filepath = {'fps': 'data/gpu_game_fps/fps_num_lower.csv'}
num_samples = 1000
bitmaps = generate_bitmap(sql_csv_file, dict_table_filepath, num_samples)
with open('data/gpu_game_fps/fps_sql_test' + '.samplebitmap', 'wb') as f:
    pickle.dump(bitmaps, f)
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

def join_order(joins):
    join_tables=[]
    for join in joins:
        join_tables.append(join.split('=')[0].split('.')[0])
        join_tables.append(join.split('=')[1].split('.')[0])
    join_tables = list(pd.value_counts(join_tables).sort(ascending = False).index)
    join_order=[]
    for t in join_tables:
        for join in joins:
            tables = [join.split('=')[0].split('.')[0], join.split('=')[1].split('.')[0]]
            if t in tables:
                join_order.append(join)
                joins.remove(join)
    return join_order


def query_on_sample(df_samples, sql_csv_line):
    joins = sql_csv_line[1].split(',')
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
        if predicates[i + 1] == '=':
            predicates[i + 1] = '=='
        df_sample[t] = df_sample[t][
            eval('df_sample[t][predicates[i].split(".")[1]]' + predicates[i + 1] + predicates[i + 2])]
        if len(df_sample[t]) == 0:
            return df_sample[t]
        i = i + 3
    if len(joins[0]) > 0:
        joins = join_order(joins)
        i = 0
        joined_t = []
        for join in joins:
            left = join.split('=')[0]
            right = join.split('=')[1]
            t_left = left.split('.')[0]
            col_left = left.split('.')[1]
            t_right = right.split('.')[0]
            col_right = right.split('.')[1]
            if i == 0:
                if col_left == col_right:
                    df_merge = df_sample[t_left].merge(df_sample[t_right], on=col_left)
                else:
                    df_merge = df_sample[t_left].merge(df_sample[t_right], left_on=col_left, right_on=col_right)
                if len(df_merge) == 0:
                    return df_merge
                joined_t.append(t_left)
                joined_t.append(t_right)
                i = i + 1
            else:
                if t_left in joined_t:
                    if col_left == col_right:
                        df_merge = df_merge.merge(df_sample[t_right], on=col_left)
                    else:
                        df_merge = df_merge.merge(df_sample[t_right], left_on=col_left, right_on=col_right)
                    if len(df_merge) == 0:
                        return df_merge
                    joined_t.append(t_right)
                elif t_right in joined_t:
                    if col_left == col_right:
                        df_merge = df_merge.merge(df_sample[t_left], on=col_left)
                    else:
                        df_merge = df_merge.merge(df_sample[t_left], left_on=col_left, right_on=col_right)
                    if len(df_merge) == 0:
                        return df_merge
                    joined_t.append(t_left)
        return df_merge
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
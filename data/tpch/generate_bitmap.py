import csv
import os
import pickle
import re
import pandas as pd
import numpy as np
import psycopg2
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


def select_sample(table_names, conn, num_samples):
    df_samples = {}
    for t in table_names:
        sql = f'SELECT * FROM {t} order by random() limit {num_samples}'
        df = pd.read_sql(sql, conn)
        df.columns = [c.lower() for c in df.columns]
        #df.reset_index(drop=True, inplace=True)
        #index = df.index
        #while len(df)<num_samples:
        #    for i in index:
        #        df = df.append(df.loc[i])
        #        df.reset_index(drop=True, inplace=True)
        #        if len(df) == num_samples:
        #            break
        df[f"bit_id_{t.split(' ')[1]}"] = [i for i in range(len(df))]
        df_samples[t.split(' ')[1]] = df

    return df_samples


def query_on_sample(df_samples, sql_csv_line):
    tables = sql_csv_line[0].split(',')
    predicates = sql_csv_line[2].split(',')
    df_sample = {}
    for t in tables:
        t = t.split(' ')[1]
        df_sample[t] = df_samples[t]
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

    return df_sample


def generate_bitmap(sql_csv_file, conn, num_samples):
    all_tables = load_sql_csv(sql_csv_file)
    table_names = get_all_table_names(all_tables)
    df_samples = select_sample(table_names, conn, num_samples)
    bitmaps = []
    with open(sql_csv_file) as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for line in tqdm(lines):
            tables = line[0].split(',')
            bitmap = np.zeros([len(tables), num_samples], dtype=np.int8)
            predicates = line[2]
            if len(predicates) == 0:
                bitmaps.append(np.ones_like(bitmap))
                continue
            df_results = query_on_sample(df_samples, line)
            t_position = 0
            for t in tables:
                t = t.split(' ')[1]
                if len(df_results[t]) == 0:
                    t_position = t_position+1
                    continue
                index = df_results[t][f"bit_id_{t}"]
                bitmap[t_position][index] = 1
                t_position = t_position+1
            bitmaps.append(bitmap)
    return bitmaps


conn = psycopg2.connect(database='tpch',
                        user='postgres', password='',
                        host='localhost', port='5432'
                        )
sql_csv_file = 'data/tpch/tpch_sql.csv'
bitmap = generate_bitmap(sql_csv_file, conn, 1000)
with open('data/tpch/tpch_sql.samplebitmap', 'wb') as f:
    pickle.dump(bitmap, f)

sql_csv_file = 'data/tpch/tpch_sql_test.csv'
bitmap = generate_bitmap(sql_csv_file, conn, 1000)
with open('data/tpch/tpch_sql_test.samplebitmap', 'wb') as f:
    pickle.dump(bitmap, f)


for i in range(0, 7):
    sql_csv_file = f'data/tpch/more_joins/tpch_sql_test_{i}.csv'
    bitmap = generate_bitmap(sql_csv_file, conn, 1000)
    with open(f'data/tpch/more_joins/tpch_sql_test_{i}.samplebitmap', 'wb') as f:
        pickle.dump(bitmap, f)
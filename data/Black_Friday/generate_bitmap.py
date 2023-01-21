import os
import pickle
import re

import pandas as pd

def prepare_samples(sql_path, samples):
    sample_bitmaps = []
    with open(sql_path + '.csv', 'r') as f:
        lines = f.readlines()
        for line in lines:
            tables = [x.split(' ')[1] for x in line.split('#')[0].split(',')]
            conds = [x for x in line.split('#')[2].split(',')]
            table2conditions = {}
            for i in range(int(len(conds) / 3)):
                t = conds[i * 3].split('.')[0]
                attr = conds[i * 3].split('.')[1]
                op = conds[i * 3 + 1]
                value = conds[i * 3 + 2]
                if t in table2conditions:
                    table2conditions[t].append((attr, op, value))
                else:
                    table2conditions[t] = [(attr, op, value)]
            sample_bitmap = []
            for table in tables:
                try:
                    # print(table2conditions)
                    # print(table)
                    # print(samples)
                    data_samples = samples[table]
                    conds = table2conditions[table]
                    bool_array = None
                    # print('conds:', conds)
                    for cond in conds:
                        # print('cond:', cond)
                        # print (table, cond)
                        attr = cond[0]
                        if cond[1] == '=':
                            barray = (data_samples[attr] == float(cond[2]))
                        elif cond[1] == '<':
                            barray = (data_samples[attr] < float(cond[2]))
                        elif cond[1] == '>':
                            barray = (data_samples[attr] > float(cond[2]))
                        else:
                            raise Exception(cond)
                        if bool_array is None:
                            bool_array = barray
                        else:
                            bool_array = bool_array & barray
                        # print('bool_array', bool_array)
                    sample_bitmap.append(bool_array.astype(int).values)  # Only single tables col6,8 are indented
                except Exception as e:
                    # f2.write('Pass '+query+'\n')
                    pass
                continue
            # print('sample_bitmap', sample_bitmap)
            sample_bitmaps.append(sample_bitmap)
    # print(sample_bitmaps)
    return sample_bitmaps


def select_samples(data_dir, table, alias):
    table2alias = {table: alias}  # modify
    # print('table2alias:', table2alias)
    samples = {}
    for table, alias in table2alias.items():
        samples[alias] = pd.read_csv(data_dir, quotechar='"', escapechar='\\',
                                     error_bad_lines=False, low_memory=False).sample(n=1000)
    return samples


min_max_file = 'data/Black_Friday/column_min_max_vals.csv'
data_dir = 'data/Black_Friday/Black_Friday_Purchase_num.csv'
table = 'black_friday_purchase'
alias = 'bfp'
sql_path = 'data/Black_Friday/black_friday_purchase_sql_test'
samples = select_samples(data_dir, table, alias)
sample_bitmaps = prepare_samples(sql_path, samples)
with open(sql_path + '.samplebitmap', 'wb') as f:
    pickle.dump(sample_bitmaps, f)

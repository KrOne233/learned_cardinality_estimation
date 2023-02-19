import pandas as pd
import numpy as np
import psycopg2
from tqdm import tqdm
import csv
from io import StringIO


table = pd.read_csv('data/Black_Friday/Black_Friday_Purchase.csv')
table.dropna(inplace=True)
for col in table.columns:
    res = dict()
    if table[col].dtype == 'object':
        data_array = np.array(table[col])
        data_list = data_array.tolist()
        data_list = list(set(data_list))
        data_list.sort()
        i = 1
        for key in data_list:
            res[key] = i
            i += 1
        for i in tqdm(range(len(table[col]))):
            if (pd.notnull(table[col].iloc[i])):
                table[col].iloc[i] = res[str(table[col].iloc[i])]
    elif table[col].dtype == 'float64':
        table[col] = table[col].astype('int64')

table.columns = [c.lower() for c in table.columns]
table.to_csv('data/Black_Friday/Black_Friday_Purchase_num.csv', index=False)

# import data into postgreSQL
table = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv')

conn = psycopg2.connect(database="black_friday_purchase",
                        user='postgres', password='',
                        host='localhost', port='5432'
)
conn.autocommit = True
cursor = conn.cursor()
sql = '''CREATE TABLE black_friday_purchase (
    id integer NOT NULL PRIMARY KEY,
    gender integer,
    age integer,
    occupation integer,
    city_category integer,
    stay_in_current_city_years integer,
    marital_status integer,
    product_category_1 integer,
    product_category_2 integer,
    product_category_3 integer,
    purchase integer 
);'''
cursor.execute(sql)

sio = StringIO()
writer = csv.writer(sio)
writer.writerows(table.values.astype('int'))
sio.seek(0)
with conn.cursor() as c:
    c.copy_from(
        file=sio,
        table="black_friday_purchase",
        columns=[
            'id',
            'gender',
            'age',
            'occupation',
            'city_category',
            'stay_in_current_city_years',
            'marital_status',
            'product_category_1',
            'product_category_2',
            'product_category_3',
            'purchase'
        ],
        sep=","
    )
    conn.commit()

# gain min_max file
table = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv')
dictalias = {'black_friday_purchase': ['bfp']}
def get_col_statistics(cols, table, alias):
    names = []
    cards = []
    distinct_nums = []
    mins = []
    maxs = []
    for col in cols:
        names.append(alias+'.'+col)
        print(col)
        maxs.append(table[col].max())
        mins.append(table[col].min())
        cards.append(len(table[col]))
        distinct_nums.append(len(table[col].unique()))
    statistics = pd.DataFrame(
        data={'name': names, 'min': mins, 'max': maxs, 'cardinality': cards, 'num_unique_values': distinct_nums})
    return statistics
    #statistics.to_csv(min_max_file, index=False)

title_statistics = get_col_statistics(table.columns, table, "bfp")
title_statistics.to_csv("data/Black_Friday/column_min_max_vals.csv", index=False)

# prepare true cardinality file and corresponding sql file for deepdb
def gen_deepdb_true_card(sql_file_csv, deepdb_file_dir, deepdb_sql_dir):
    with open(sql_file_csv,'r') as f:
        lines = f.readlines()
        with open(deepdb_file_dir +'/deepdb_true_cardinalities.csv','w') as d:
            d.write('query_no,query,cardinality_true\n')
            with open(deepdb_sql_dir + '/deepdb_sql.sql','w') as s:
                for no_query, query_str in enumerate(lines):
                    query_str = query_str.split("#")
                    tables = query_str[0]
                    joins = query_str[1]
                    predicates = query_str[2].split(',')
                    card = query_str[3]
                    if len(joins) == 0:
                        qstr = "SELECT COUNT(*) FROM " + str(tables) + " WHERE "
                    else:
                        qstr = "SELECT COUNT(*) FROM " + str(tables) + " WHERE "
                        joins = joins.split(',')
                        for j in range(len(joins)):
                            qstr += str(joins[j]) + ' AND '
                    i = 0
                    p = ''
                    while i < len(predicates):
                        p = p + str(predicates[i]) + str(predicates[i+1]) + str(predicates[i+2]) + ' AND '
                        i = i + 3
                    qstr = qstr + p[:len(p) - 5] + ';'
                    d.write(f'{no_query},"{qstr}",{card}')
                    s.write(f'{qstr}\n')
            s.close()
        d.close()
    f.close()

gen_deepdb_true_card('data/Black_Friday/black_friday_purchase_sql_test.csv', 'data/Black_Friday', 'data/Black_Friday')


# for 2_col_test
def gen_deepdb_true_card(sql_file_csv, deepdb_file_dir, deepdb_sql_dir,i):
    with open(sql_file_csv,'r') as f:
        lines = f.readlines()
        with open(deepdb_file_dir +f'/deepdb_true_cardinalities_2col_{i}.csv','w') as d:
            d.write('query_no,query,cardinality_true\n')
            with open(deepdb_sql_dir + f'/deepdb_sql_2col_{i}.sql','w') as s:
                for no_query, query_str in enumerate(lines):
                    query_str = query_str.split("#")
                    tables = query_str[0]
                    joins = query_str[1]
                    predicates = query_str[2].split(',')
                    card = query_str[3]
                    if len(joins) == 0:
                        qstr = "SELECT COUNT(*) FROM " + str(tables) + " WHERE "
                    else:
                        qstr = "SELECT COUNT(*) FROM " + str(tables) + " WHERE "
                        joins = joins.split(',')
                        for j in range(len(joins)):
                            qstr += str(joins[j]) + ' AND '
                    i = 0
                    p = ''
                    while i < len(predicates):
                        p = p + str(predicates[i]) + str(predicates[i+1]) + str(predicates[i+2]) + ' AND '
                        i = i + 3
                    qstr = qstr + p[:len(p) - 5] + ';'
                    d.write(f'{no_query},"{qstr}",{card}')
                    s.write(f'{qstr}\n')
            s.close()
        d.close()
    f.close()

i = 1
deepdb_file_dir = 'data/Black_Friday/2_col_sql'
deepdb_sql_dir = 'data/Black_Friday/2_col_sql'
while i<8:
    sql_file_csv = f'data/Black_Friday/2_col_sql/bfp_2col_{i}_test.csv'
    gen_deepdb_true_card(sql_file_csv, deepdb_file_dir, deepdb_sql_dir,i)
    i=i+2







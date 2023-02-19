import csv
import datetime
from io import StringIO
import numpy as np
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine
import psycopg2
from numpy.random import choice
import datetime


def import_postgre(table_name, db_name, df):
    engine = create_engine(f'postgres://postgres:wzy07wx25@localhost:5432/{db_name}')
    pd_engine = pd.io.sql.pandasSQL_builder(engine)
    table = pd.io.sql.SQLTable(table_name, pd_engine, frame=df, index=False, if_exists='fail')
    table.create()
    io_buff = StringIO()
    df.to_csv(io_buff, sep='\t', index=False, header=False)
    io_buff_value = io_buff.getvalue()
    conn = psycopg2.connect(database=db_name,
                            user='postgres', password='',
                            host='localhost', port='5432'
                            )
    cur = conn.cursor()
    cur.copy_from(StringIO(io_buff_value), table_name, null='')
    conn.commit()
    cur.close()
    conn.close()


with open("data/tpch/lineitem.tbl", 'r') as f:
    lines = f.readlines()
    with open('data/tpch/lineitem1.tbl', 'w') as w:
        for line in lines:
            w.write(line[:-2] + '\n')

lineitem = pd.read_csv("data/tpch/lineitem1.tbl", sep='|', header=None)
import_postgre('lineitem', 'tpch', lineitem)

# used column in each table
lineitem_usecols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice',
                    'l_discount', 'l_tax']
customer_usecols = ['c_custkey', 'c_nationkey', 'c_acctbal']

nation_usecols = ['n_nationkey', 'n_regionkey']

orders_usecols = ['o_orderkey', 'o_custkey', 'o_totalprice']  # 'o_orderdate'

part_usecols = ['p_partkey', 'p_size', 'p_retailprice']

partsupp_usecols = ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost']

region_usecols = ['r_regionkey']  # 这个表总共5行，3列，剩下两列是名字和comments

supplier_usecols = ['s_suppkey', 's_nationkey', 's_acctbal']

# generate join key for ps and l
lineitem = pd.read_csv('data/tpch/lineitem.csv', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"')
partsupp = pd.read_csv('data/tpch/partsupp.csv', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"')

partsupp['ps_id'] = range(len(partsupp))
dict = {}
for i in tqdm(range(len(partsupp))):
    dict[(partsupp['ps_partkey'][i], partsupp['ps_suppkey'][i])] = i

lineitem['l_ps_id'] = range(len(lineitem))
for j in tqdm(range(len(lineitem))):
    lineitem['l_ps_id'][j] = dict[(lineitem['l_partkey'][j], lineitem['l_suppkey'][j])]

partsupp.to_csv('data/tpch/partsupp2.csv', escapechar='\\', quotechar='"', index=False, encoding='utf-8')
lineitem.to_csv('data/tpch/lineitem2.csv', escapechar='\\', quotechar='"', index=False, encoding='utf-8')

import_postgre('lineitem', 'tpch', lineitem)
import_postgre('partsupp', 'tpch', partsupp)

# generate single primary key for lineitem
lineitem = pd.read_csv('data/tpch/lineitem.csv', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"')
lineitem['l_key'] = range(len(lineitem))
import_postgre('lineitem', 'tpch', lineitem)
lineitem.to_csv('data/tpch/lineitem.csv', escapechar='\\', quotechar='"', index=False, encoding='utf-8')

# method deal with date
conn = psycopg2.connect(database='tpch',
                        user='postgres', password='',
                        host='localhost', port='5432'
                        )
cur = conn.cursor()
conn.autocommit = True
sql = 'SELECT o_orderdate FROM orders o order by random() limit 1'
cur.execute(sql)
val = cur.fetchall()[0][0]
if type(val) == datetime.date:
    val = f"'{val}'"
sql = 'SELECT o_orderdate FROM orders o WHERE o.o_orderdate>'
sql = sql + val
cur.execute(sql)


# gain min_max file
def get_col_statistics(cols, table, alias):
    names = []
    cards = []
    distinct_nums = []
    mins = []
    maxs = []
    for col in cols:
        names.append(alias + '.' + col)
        print(col)
        maxs.append(table[col].max())
        mins.append(table[col].min())
        cards.append(len(table[col]))
        distinct_nums.append(len(table[col].unique()))
    statistics = pd.DataFrame(
        data={'name': names, 'min': mins, 'max': maxs, 'cardinality': cards, 'num_unique_values': distinct_nums})
    return statistics


conn = psycopg2.connect(database='tpch',
                        user='postgres', password='',
                        host='localhost', port='5432'
                        )

lineitem = pd.read_sql('SELECT * FROM lineitem', conn)[lineitem_usecols]
l_st = get_col_statistics(lineitem.columns, lineitem, 'l')
del lineitem

customer = pd.read_sql('SELECT * FROM customer', conn)[customer_usecols]
c_st = get_col_statistics(customer.columns, customer, 'c')
del customer

nation = pd.read_sql('SELECT * FROM nation', conn)[nation_usecols]
n_st = get_col_statistics(nation.columns, nation, 'n')
del nation

orders = pd.read_sql('SELECT * FROM orders', conn)[orders_usecols]
o_st = get_col_statistics(orders.columns, orders, 'o')
del orders

part = pd.read_sql('SELECT * FROM part', conn)[part_usecols]
p_st = get_col_statistics(part.columns, part, 'p')
del part

partsupp = pd.read_sql('SELECT * FROM partsupp', conn)[partsupp_usecols]
ps_st = get_col_statistics(partsupp.columns, partsupp, 'ps')
del partsupp

region = pd.read_sql('SELECT * FROM region', conn)[region_usecols]
r_st = get_col_statistics(region.columns, region, 'r')
del region

supplier = pd.read_sql('SELECT * FROM supplier', conn)[supplier_usecols]
s_st = get_col_statistics(supplier.columns, supplier, 's')
del supplier

statistic = pd.concat([l_st, c_st, n_st, o_st, p_st, ps_st, r_st, s_st])

statistic.to_csv("data/tpch/column_min_max_vals.csv", index=False, encoding='utf-8')


# prepare deepdb qeury file
def gen_deepdb_true_card(sql_file_csv, deepdb_file_dir, deepdb_sql_dir):
    with open(sql_file_csv, 'r') as f:
        lines = f.readlines()
        with open(deepdb_file_dir + '/deepdb_true_cardinalities.csv', 'w') as d:
            d.write('query_no,query,cardinality_true\n')
            with open(deepdb_sql_dir + '/deepdb_sql.sql', 'w') as s:
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

                    if len(predicates[0]) > 0:
                        i = 0
                        p = ''
                        while i < len(predicates):
                            p = p + str(predicates[i]) + str(predicates[i + 1]) + str(predicates[i + 2]) + ' AND '
                            i = i + 3
                        qstr = qstr+p
                    qstr = qstr[:len(qstr) - 5] + ';'
                    d.write(f'{no_query},"{qstr}",{card}')
                    s.write(f'{qstr}\n')
            s.close()
        d.close()
    f.close()

gen_deepdb_true_card('data/tpch/more_joins/tpch_sql_test_6.csv', 'data/tpch/more_joins', 'data/tpch/more_joins')


# adjust queries with table filtered by primary key
l_usecols = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice',
             'l_discount', 'l_tax']
c_usecols = ['c_nationkey', 'c_acctbal']

n_usecols = ['n_regionkey']

o_usecols = ['o_custkey', 'o_totalprice']  # 'o_orderdate'

p_usecols = ['p_size', 'p_retailprice']

ps_usecols = ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost']

r_usecols = []  # 这个表总共5行，3列，主键外剩下两列是名字和comments

s_usecols = ['s_nationkey', 's_acctbal']

primary_keys = ['l_key', 'c_custkey', 'n_nationkey', 'o_orderkey', 'p_partkey', 'ps_id', 'r_regionkey', 's_suppkey']

conn = psycopg2.connect(database='tpch',
                        user='postgres', password='',
                        host='localhost', port='5432'
                        )
cur = conn.cursor()
conn.autocommit = True

with open('data/tpch/more_joins/tpch_sql_test_4.csv', 'r') as f:
    lines = f.readlines()
    ops = ['<', '>', '=']
    with open('data/tpch/more_joins/tpch_sql_test_4_adj.csv', 'w') as n:
        for line in tqdm(lines):
            q = line.split('#')
            predicates = q[2].split(',')
            if len(predicates[0]) > 0:
                joins = q[1].split(',')
                if len(joins[0]) > 0:
                    sql = f'SELECT COUNT(*) FROM {q[0]} WHERE {" AND ".join(joins)} AND '
                else:
                    sql = f'SELECT COUNT(*) FROM {q[0]} WHERE '
                i = 0
                new_col = []
                new_predicates = []
                while i < len(predicates):
                    col = predicates[i]
                    t = col.split('.')[0]
                    if col.split('.')[1] in primary_keys:
                        #predicates.remove(predicates[i])
                        #predicates.remove(predicates[i+1])
                        #predicates.remove(predicates[i+2])
                        if len(eval(f'{t}_usecols'))>0:
                            new_col.append(f'{t}.' + choice(eval(f'{t}_usecols')))
                    else:
                        sql = sql + predicates[i] + predicates[i+1] + str(predicates[i+2]) + ' AND '
                        new_predicates.append(predicates[i])
                        new_predicates.append(predicates[i + 1])
                        new_predicates.append(str(predicates[i + 2]))
                    i = i+3

                if len(new_predicates)>0:
                    if new_predicates == predicates:
                        n.write(line)
                        continue
                    else:
                        cur.execute(sql[:-5])
                        card = cur.fetchall()[0][0]
                        n.write('#'.join(q[:-2]) + '#' + ','.join(new_predicates) + '#' + str(card) + '\n')
                else:
                    if len(new_col) == 0:
                        continue
                    else:
                        sql = sql.replace('COUNT(*)', ','.join(new_col))
                        if len(joins[0]) > 0:
                            df = pd.read_sql(sql[:-5], conn)
                        else:
                            df = pd.read_sql(sql[:-7], conn)

                        for col in new_col:
                            card = len(df)
                            op = choice(ops)
                            val = choice(df[col.split('.')[1]].unique())
                            if op == '=':
                                op_df = '=='
                            df = df[eval(f'df[col.split(".")[1]]{op_df}{str(val)}')]
                            if len(df) == 0:
                                break
                            else:
                                # new_p = col + op + str(val)
                                new_predicates.append(col)
                                new_predicates.append(op)
                                new_predicates.append(str(val))
                                card = len(df)

                        n.write('#'.join(q[:-2]) + '#' + ','.join(new_predicates) + '#' + str(card) + '\n')
            else:
                n.write(line)

'''
with open('data/tpch/tpch_sql_adj.csv', 'r') as f:
    lines = f.readlines()
    with open('data/tpch/tpch_sql_adj2.csv', 'w') as n:
        for line in tqdm(lines):
            q = line.split("#")
            predicates = q[2].split(',')
            new_p = []
            if len(predicates[0])>0:
                for p in predicates:
                    if len(p.split('>')) > 1 and p != '>':
                        new_p.append(p.split('>')[0])
                        new_p.append('>')
                        new_p.append(p.split('>')[1])
                    elif len(p.split('=')) > 1 and p != '=':
                        new_p.append(p.split('=')[0])
                        new_p.append('=')
                        new_p.append(p.split('=')[1])
                    elif len(p.split('<')) > 1 and p != '<':
                        new_p.append(p.split('<')[0])
                        new_p.append('<')
                        new_p.append(p.split('<')[1])
                    else:
                        new_p.append(p)
                n.write(q[0] + '#' + q[1] + '#' + ','.join(new_p) + '#' + q[3])
            else:
                n.write(line)

'''

# just remove the query that has primary key in predicates
with open('data/tpch/more_joins/tpch_sql_test_6.csv', 'r') as f:
    lines = f.readlines()
    with open('data/tpch/more_joins/tpch_sql_test_6_reduced.csv', 'w') as n:
        for line in tqdm(lines):
            q = line.split('#')
            predicates = q[2].split(',')
            if len(predicates[0]) > 0:
                i = 0
                while i < len(predicates):
                    col = predicates[i]
                    t = col.split('.')[0]
                    if col.split('.')[1] in primary_keys:
                        line = []
                        break
                    i = i+3
            if len(line) == 0:
                continue
            else:
                n.write(line)


with open('data/tpch/tpch_sql_test.csv','r') as f:
    lines = f.readlines()
    with open('data/tpch/more_joins/tpch_sql_test_2.csv.csv', 'w') as w:
        for line in lines:
            q = line.split('#')
            tables = q[0].split(',')
            if len(tables) == 3:
                w.write(line)
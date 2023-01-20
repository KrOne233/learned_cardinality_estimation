import random
from numpy.random import choice
import pandas as pd
import numpy as np
import psycopg2
from tqdm import tqdm
from sqlalchemy import create_engine
from io import StringIO

df = pd.read_csv('data/gpu_game_fps/gpu_game_fps.csv', escapechar='\\', encoding='utf-8',
                 low_memory=False, quotechar='"')

for col in df.columns:
    # replace na with 0
    if df[col].isna().any():
        df[col].fillna(0, inplace=True)
    # transform categorical value to numeric
    if df[col].dtype == 'object':
        data = set(df[col])
        dic = {}
        i = 1
        for d in data:
            dic[d] = i
            i = i + 1
        for r in tqdm(range(len(df[col]))):
            df[col][r] = dic[df[col][r]]
    # 把浮点型式的整数规范为整数型
    elif df[col].dtype == 'float64':
        data = np.array(list(set(df[col])))
        if sum(data - np.floor(data)) == 0:
            df[col] = df[col].astype('int64')

df.to_csv('data/gpu_game_fps/fps_num.csv')

df = pd.read_csv('data/gpu_game_fps/fps_num.csv', escapechar='\\', encoding='utf-8',
                 low_memory=False, quotechar='"')

df.columns[0] = 'id'
df.columns = [x.lower().replace(' ', '') for x in list(df.columns)]

df.to_csv('data/gpu_game_fps/fps_num_lower.csv', index=False, encoding='utf-8')

# import table to postgresql
engine = create_engine('postgres://postgres:wzy07wx25@localhost:5432/fps')
pd_engine = pd.io.sql.pandasSQL_builder(engine)
table = pd.io.sql.SQLTable('fps', pd_engine, frame=df, index=False, if_exists='fail')
table.create()
io_buff = StringIO()
df.to_csv(io_buff, sep='\t', index=False, header=False)
io_buff_value = io_buff.getvalue()
conn = psycopg2.connect(database="fps",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
                        )
cur = conn.cursor()
cur.copy_from(StringIO(io_buff_value), 'fps')
conn.commit()
cur.close()
conn.close()

# generate sql
df_fps = pd.read_csv('data/gpu_game_fps/fps_num_lower.csv', escapechar='\\', encoding='utf-8',
                     low_memory=False, quotechar='"',
                     usecols=['id', 'cpunumberofcores', 'cpunumberofthreads',
                              'cpubaseclock', 'cpucachel1', 'cpucachel2', 'cpucachel3', 'cpudiesize',
                              'cpufrequency', 'cpumultiplier', 'cpumultiplierunlocked',
                              'cpuprocesssize', 'cputdp', 'cpunumberoftransistors', 'cputurboclock',
                              'gpubandwidth', 'gpubaseclock', 'gpuboostclock', 'gpubusnterface',
                              'gpunumberofcomputeunits', 'gpudiesize', 'gpunumberofexecutionunits',
                              'gpufp32performance', 'gpumemorybus', 'gpumemorysize', 'gpumemorytype', 'gpupixelrate',
                              'gpuprocesssize', 'gpunumberofrops', 'gpushadermodel', 'gpunumberofshadingunits',
                              'gpunumberoftmus', 'gputexturerate', 'gpunumberoftransistors',
                              'gpuvulkan', 'gamename', 'gameresolution', 'gamesetting', 'fps'])

dictalias = {'fps': ['f']}

conn = psycopg2.connect(database="fps",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
                        )
conn.autocommit = True
cursor = conn.cursor()

t_col = list(df_fps.columns)[1:]  # id is not used for filtering
ops = ['=', '<', '>']  # operations
predicates = []
tables = ['fps f']
joins = []
f = open("data/gpu_game_fps/fps_sql.csv", 'w')
f_sql = open("data/gpu_game_fps/fps.sql", 'w')
for i in tqdm(range(40000)):
    questr = 'SELECT COUNT(*) FROM '
    questr = questr + ",".join(tables) + " WHERE "
    num_col = random.randint(1, 10)  # number of columns
    col = list(choice(t_col, num_col, replace=False))
    component = []
    result = []
    for k in range(num_col):
        component.append(dictalias['fps'][0] + '.' + str(col[k]))
        op = choice(ops)
        component.append(op)
        val = int(df_fps[col[k]][random.randint(0, len(df_fps[col[k]]) - 1)])
        component.append(val)
        questr += dictalias['fps'][0] + '.' + str(col[k]) + op + str(val) + ' AND '
    questr = questr[:len(questr) - 5]
    questr += ';'
    # df = pd.read_sql(questr, conn)
    # card = df['count'].values[0]
    cursor.execute(questr)
    # card = len(cursor.fetchall())
    card = cursor.fetchall()[0][0]
    questr += f',{card}\n'
    predicates.append(component)
    f.write(",".join(tables) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
    f_sql.write(questr)
f.close()
f_sql.close()

# generate train and testset
with open("data/gpu_game_fps/fps_sql.csv", "r") as input:
    lines = input.readlines()
    with open("data/gpu_game_fps/fps_sql_train.csv", "w") as output_train:
        with open("data/gpu_game_fps/fps_sql_test.csv", "w") as output_test:
            i = 0
            testlines = random.sample(range(len(lines)), 1000)
            for line in lines:
                if i in testlines:
                    output_test.write(line)
                else:
                    output_train.write(line)
                i = i+1
        output_test.close()
    output_train.close()
input.close()


# prepare files for deepdb
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
                        qstr = "SELECT COUNT(*) FROM " + str(tables) + " WHERE " + str(joins) + ' AND '
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

gen_deepdb_true_card('data/gpu_game_fps/fps_sql_test.csv', 'data/gpu_game_fps', 'data/gpu_game_fps')

# get min_max_file
table = pd.read_csv('data/gpu_game_fps/fps_num_lower.csv')
dictalias = {'fps': ['f']}
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

title_statistics = get_col_statistics(table.columns, table, "f")
title_statistics.to_csv("data/gpu_game_fps/column_min_max_vals.csv", index=False)
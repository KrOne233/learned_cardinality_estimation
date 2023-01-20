import csv
from io import StringIO
import numpy as np
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import psycopg2
import re

# transform categorical data to numeric,clean data
df = pd.read_csv('data/imdb/title.csv', sep=',', escapechar='\\', encoding='utf-8',
                 low_memory=False, quotechar='"')
usecols = ['imdb_index', 'kind_id', 'production_year', 'imdb_id',
           'phonetic_code', 'episode_of_id', 'season_nr', 'episode_nr',
           'series_years']

dic_col2dist = {}
for col in usecols:
    if df[col].dtype == 'object':
        data = set(df[col].dropna())
        dic = {}
        i = 1
        for d in data:
            dic[d] = i
            i = i + 1
    dic_col2dist[col] = dic

COMMA_MATCHER = re.compile(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")

with open('data/imdb/title.csv','r') as f:
    lines = f.readlines()
    dic_col = {}
    i = 0
    for col in lines[0].removesuffix('\n').split(','):
        dic_col[i] = col
        i = i+1
    with open('data/imdb/title_cleaned.csv','w') as w:
        w.write(lines[0])
        for line in tqdm(lines[1:]):
            i = 0
            str_line = ''
            for val in COMMA_MATCHER.split(line.removesuffix('\n')):
                if dic_col[i] in usecols:
                    if len(val) > 0:
                        if df[dic_col[i]].dtype == 'object':
                            val = dic_col2dist[dic_col[i]][val]
                        else:
                            if float(val)-np.floor(float(val))==0:  # 将小数型的整数规范为整数
                                val = str(int(val))
                    else:
                        val = '0'  # replace na with 0
                str_line = str_line+str(val)+','
                i = i+1
            w.write(str_line.removesuffix(',')+'\n')
    w.close()
f.close()


''' # too slow
def clean(df, usecols):
    for col in df.columns:
        # replace na with 0
        if df[col].isna().any():
            df[col].fillna(0, inplace=True)
        # transform categorical value to numeric
        if col in usecols:
            if df[col].dtype == 'object':
                data = set(df[col].dropna())
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
    return df

df_cleaned = clean(df, usecols)
df.cleaned.to_csv('data/imdb/title_cleaned.csv', index=False, encoding='utf-8')
'''



# import table to database
engine = create_engine('postgres://postgres:wzy07wx25@localhost:5432/imdb')
pd_engine = pd.io.sql.pandasSQL_builder(engine)
df_title = pd.read_csv('data/imdb/title_cleaned.csv', sep=',', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"')
table = pd.io.sql.SQLTable('title', pd_engine, frame=df_title, index=False, if_exists='fail')
table.create()
io_buff = StringIO()
df_title.to_csv(io_buff, sep='\t', index=False, header=False)
io_buff_value = io_buff.getvalue()
conn = psycopg2.connect(database="imdb",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
                        )
cur = conn.cursor()
cur.copy_from(StringIO(io_buff_value), 'title', null='')
conn.commit()
cur.close()
conn.close()

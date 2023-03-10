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

with open('data/imdb/title.csv', 'r') as f:
    lines = f.readlines()
    dic_col = {}
    i = 0
    for col in lines[0].removesuffix('\n').split(','):
        dic_col[i] = col
        i = i + 1
    with open('data/imdb/title_cleaned.csv', 'w') as w:
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
                            if float(val) - np.floor(float(val)) == 0:  # 将小数型的整数规范为整数
                                val = str(int(val))
                    else:
                        val = '0'  # replace na with 0
                str_line = str_line + str(val) + ','
                i = i + 1
            w.write(str_line.removesuffix(',') + '\n')
    w.close()
f.close()

# import table to database
df_title = pd.read_csv('data/imdb/title_cleaned.csv', sep=',', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"')  # 用 MSCN
df_movie_info_idx = pd.read_csv('data/imdb/movie_info_idx.csv', sep=',', escapechar='\\', encoding='utf-8',
                                low_memory=False, quotechar='"')  # 用
df_movie_info = pd.read_csv('data/imdb/movie_info.csv', sep=',', escapechar='\\', encoding='utf-8',
                            low_memory=False, quotechar='"', usecols=['id', 'movie_id', 'info_type_id'])# 用

df_info_type = pd.read_csv('data/imdb/info_type.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"')
df_cast_info = pd.read_csv('data/imdb/cast_info.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"', usecols=['id', 'movie_id', 'person_id', 'role_id'])  # 用
df_char_name = pd.read_csv('data/imdb/char_name.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"')
df_role_type = pd.read_csv('data/imdb/role_type.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"')
df_complete_cast = pd.read_csv('data/imdb/complete_cast.csv', sep=',', escapechar='\\', encoding='utf-8',
                               low_memory=False, quotechar='"')
df_comp_cast_type = pd.read_csv('data/imdb/comp_cast_type.csv', sep=',', escapechar='\\', encoding='utf-8',
                                low_memory=False, quotechar='"')
df_name = pd.read_csv('data/imdb/name.csv', sep=',', escapechar='\\', encoding='utf-8',
                      low_memory=False, quotechar='"')  # 人名，有na，更多信息关于人（比如性别）
df_aka_name = pd.read_csv('data/imdb/aka_name.csv', sep=',', escapechar='\\', encoding='u1tf-8',
                          low_memory=False, quotechar='"')  # 人名，有na，行少很多
df_movie_keyword = pd.read_csv('data/imdb/movie_keyword.csv', sep=',', escapechar='\\', encoding='utf-8',
                               low_memory=False, quotechar='"')  # MSCN 用了
df_keyword = pd.read_csv('data/imdb/keyword.csv', sep=',', escapechar='\\', encoding='utf-8',
                         low_memory=False, quotechar='"')  # 不用
df_person_info = pd.read_csv('data/imdb/person_info.csv', sep=',', escapechar='\\', encoding='utf-8',
                             low_memory=False, quotechar='"')  # 想用
df_movie_companies = pd.read_csv('data/imdb/movie_companies.csv', sep=',', escapechar='\\', encoding='utf-8',
                                 low_memory=False, quotechar='"')  # 用
df_company_name = pd.read_csv('data/imdb/company_name.csv', sep=',', escapechar='\\', encoding='utf-8',
                              low_memory=False, quotechar='"')  # 想用
df_company_type = pd.read_csv('data/imdb/company_type.csv', sep=',', escapechar='\\', encoding='utf-8',
                              low_memory=False, quotechar='"')
df_aka_title = pd.read_csv('data/imdb/aka_title.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"')
df_kind_type = pd.read_csv('data/imdb/kind_type.csv', sep=',', escapechar='\\', encoding='utf-8',
                           low_memory=False, quotechar='"')


def import_postgre(table_name, db_name, df):
    engine = create_engine(f'postgres://postgres:xxx@localhost:5432/{db_name}')
    pd_engine = pd.io.sql.pandasSQL_builder(engine)
    table = pd.io.sql.SQLTable(table_name, pd_engine, frame=df, index=False, if_exists='fail')
    table.create()
    io_buff = StringIO()
    df.to_csv(io_buff, sep='\t', index=False, header=False)
    io_buff_value = io_buff.getvalue()
    conn = psycopg2.connect(database=db_name,
                            user='postgres', password='xxx',
                            host='localhost', port='5432'
                            )
    cur = conn.cursor()
    cur.copy_from(StringIO(io_buff_value), table_name, null='')
    conn.commit()
    cur.close()
    conn.close()

dict2 = {'title': ['t', 'id', 'kind_id', 'production_year', 'phonetic_code', 'season_nr', 'episode_nr'],
         'movie_info_idx': ['mi_idx', 'movie_id', 'info_type_id'],
         'movie_info': ['mi', 'movie_id', 'info_type_id'],
         'cast_info': ['ci', 'movie_id', 'person_id', 'role_id'],
         'movie_keyword': ['mk', 'movie_id', 'keyword_id'],
         'movie_companies': ['mc', 'movie_id', 'company_id', 'company_type_id']}

import_postgre('title', 'imdb', df_title)

import_postgre('movie_info_idx', 'imdb', df_movie_info_idx)

import_postgre('movie_info', 'imdb', df_movie_info)

import_postgre('cast_info', 'imdb', df_cast_info)

import_postgre('movie_keyword', 'imdb', df_movie_keyword)

import_postgre('movie_companies', 'imdb', df_movie_companies)


# select the queries with certain number of joins
with open('data/imdb/imdb_test_sql.csv','r') as f:
    lines = f.readlines()
    with open('data/imdb/more_joins/imdb_sql_test_0.csv', 'w') as w:
        for line in lines:
            q = line.split('#')
            tables = q[0].split(',')
            if len(tables) == 1:
                w.write(line)



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

gen_deepdb_true_card('data/imdb/more_joins/imdb_sql_test_4.csv', 'data/imdb/more_joins', 'data/imdb/more_joins')

# delete the queries contain foreign key in predicates
with open('data/imdb/more_joins/imdb_sql_test_3.csv','r') as f:
    lines = f.readlines()
    with open('data/imdb/more_joins/imdb_sql_test_3_adj.csv', 'w') as w:
        for line in lines:
            q = line.split('#')
            tables = q[0].split(',')
            predicates = q[2].split(',')
            if len(predicates[0])>0:
                i = 0
                while i<len(predicates):
                    if predicates[i].split('.')[1] == 'movie_id':
                        i = 100
                        break
                    i = i+3
                if i < 100:
                    w.write(line)
            else:
                w.write(line)

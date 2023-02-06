import csv
from io import StringIO
import numpy as np
from tqdm import tqdm
import pandas as pd
from sqlalchemy import create_engine
import pymysql
import psycopg2

#Transfer the string columns in title to numeric, only useful columns are read, NAs are all dropped
df_title = pd.read_csv('data/imdb/title.csv', sep=',', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"',
                       usecols=['id', 'kind_id', 'production_year', 'phonetic_code', 'season_nr', 'episode_nr'])
df_title = df_title.dropna(axis=0, how='any', inplace=False)

res = dict()
df_title.head()

data_array = np.array(df_title['phonetic_code'])
data_list = data_array.tolist()
data_list = list(set(data_list))
if np.nan in data_list:
    data_list.remove(np.nan)
data_list = [str(i) for i in data_list]
data_list.sort()
i = 1
for key in data_list:
    res[key] = i
    i += 1
for i in tqdm(range(len(df_title['phonetic_code']))):
    if (pd.notnull(df_title['phonetic_code'].iloc[i])):
        df_title['phonetic_code'].iloc[i] = res[str(df_title['phonetic_code'].iloc[i])]
'''
data_array = np.array(df['series_years'])
data_list = data_array.tolist()
data_list = list(set(data_list))
if np.nan in data_list:
    data_list.remove(np.nan)
data_list = [str(i) for i in data_list]
data_list.sort()
i = 1
for key in data_list:
    res[key] = i
    i += 1
for i in tqdm(range(len(df['series_years']))):
    if (pd.notnull(df['series_years'][i])):
        df['series_years'][i] = res[df['series_years'][i]]

data_array = np.array(df['imdb_index'])
data_list = data_array.tolist()
data_list = list(set(data_list))
if np.nan in data_list:
    data_list.remove(np.nan)
data_list = [str(i) for i in data_list]
data_list.sort()
i = 1
for key in data_list:
    res[key] = i
    i += 1
for i in tqdm(range(len(df['imdb_index']))):
    if (pd.notnull(df['imdb_index'][i])):
        df['imdb_index'][i] = res[df['imdb_index'][i]]
'''
for col in df_title.columns:
    if df_title[col].dtype == 'float64':
        df_title[col] = df_title[col].astype('int64')

df_title.to_csv("data/imdb/title_reduced_num.csv", header=True, index=False)


# upload data to PostgreSQL database
df_title = pd.read_csv("data/imdb/title_reduced_num.csv", low_memory=False)

conn = psycopg2.connect(database="Master_thesis",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
)
conn.autocommit = True
cursor = conn.cursor()
sql = '''CREATE TABLE title (
    id integer NOT NULL PRIMARY KEY,
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code integer,
    season_nr integer,
    episode_nr integer
);'''
cursor.execute(sql)

sio = StringIO()
writer = csv.writer(sio)
writer.writerows(df_title.values.astype('int'))
sio.seek(0)
with conn.cursor() as c:
    c.copy_from(
        file=sio,
        table="title",
        columns=[
            "id",
            "kind_id",
            "production_year",
            "phonetic_code",
            "season_nr",
            "episode_nr"
        ],
        sep=","
    )
    conn.commit()


engine = create_engine('postgres://postgres:xxx@localhost:5432/imdb')
pd_engine = pd.io.sql.pandasSQL_builder(engine)
df_title = pd.read_csv('data/imdb/title.csv', sep=',', escapechar='\\', encoding='utf-8',
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





# gain min_max file
df_title = pd.read_csv("data/imdb/title_reduced_num.csv", low_memory=False)
dictalias = {'title': ['t'],
             'movie_info_idx': ['mi_idx'],
             'movie_info': ['mi'],
             'cast_info': ['ci'],
             'movie_keyword': ['mk'],
             'movie_companies': ['mc']}
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

title_statistics = get_col_statistics(df_title.columns, df_title, "t")
title_statistics.to_csv("data/imdb/column_min_max_vals.csv", index=False)
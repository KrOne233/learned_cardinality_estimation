import random
import psycopg2
from tqdm import tqdm
import pandas as pd
from numpy.random import choice

conn = psycopg2.connect(database='imdb',
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
                        )
cur = conn.cursor()
conn.autocommit = True

# used column in each table
title_usecols = ['id', 'kind_id', 'production_year']  # , 'phonetic_code', 'season_nr', 'episode_nr']

movie_info_idx_usecols = ['id', 'movie_id', 'info_type_id']

movie_info_usecols = ['id', 'movie_id', 'info_type_id']

cast_info_usecols = ['id', 'movie_id', 'person_id', 'role_id']

movie_keyword_usecols = ['id', 'movie_id', 'keyword_id']

movie_companies_usecols = ['id', 'movie_id', 'company_id', 'company_type_id']

dictalias = {'t': 'title t',
             'mi_idx': 'movie_info_idx mi_idx',
             'mi': 'movie_info mi',
             'ci': 'cast_info ci',
             'mk': 'movie_keyword mk',
             'mc': 'movie_companies mc'
             }

ops = ['=', '<', '>']  # operations

tables = ['title t', 'movie_info_idx mi_idx', 'movie_info mi', 'cast_info ci', 'movie_keyword mk', 'movie_companies mc']

joins_keys = {('t', 'mi_idx'): 't.id=mi_idx.movie_id', ('t', 'mi'): 't.id=mi.movie_id',
              ('t', 'ci'): 't.id=ci.movie_id', ('t', 'mk'): 't.id=mk.movie_id',
              ('t', 'mc'): 't.id=mc.movie_id'}

join_route = {'t': ['mi_idx', 'mi', 'ci', 'mk', 'mc'], 'mi_idx': ['t'], 'mi': ['t'], 'ci': ['t'],
              'mk': ['t'], 'mc': ['t']}

f = open("data/imdb/imdb_sql.csv", 'w')
f_sql = open("data/imdb/imdb.sql", 'w')
for i in tqdm(range(10)):
    num_tables = random.randint(1, 3)  # decide tables involved
    if num_tables > 1:
        i = 1
        initial = choice(tables)
        join_order = []
        join_order.append(initial)
        while i < num_tables:
            candidates = []
            for table in join_order:
                for t in join_route[table.split(' ')[1]]:
                    candidates.append(t)
            candidates = list(set(candidates))
            choose_list = []
            for c in candidates:
                if dictalias[c] not in join_order:
                    choose_list.append(dictalias[c])
            join_order.append(choice(choose_list))
            i = i + 1

        joins = []  # decide joins
        try:
            join = joins_keys[join_order[0].split(' ')[1], join_order[1].split(' ')[1]]
        except Exception:
            join = joins_keys[join_order[1].split(' ')[1], join_order[0].split(' ')[1]]

        joins.append(join)

        n = 2
        while n < len(join_order):
            t = join_order[n].split(' ')[1]
            for table in join_order[:n]:
                if (table.split(' ')[1], t) in joins_keys.keys():
                    joins.append(joins_keys[(table.split(' ')[1], t)])
                    break
                elif (t, table.split(' ')[1]) in joins_keys.keys():
                    joins.append(joins_keys[t, table.split(' ')[1]])
                    break
                else:
                    continue
            n = n + 1

        col_key = []  # decide columns
        for key in joins:
            for k in key.split('='):
                col_key.append(k.split('.')[1])
        cols = []
        for table in join_order:
            col = eval(f'{table.split(" ")[0]}_usecols')
            for c in col:
                if c not in col_key:
                    cols.append(c)

        if len(cols) <= 4:
            num_col = random.randint(0, len(cols))
        else:
            num_col = random.randint(0, 5)  # number of columns

    else:
        join_order = [choice(tables)]
        joins = []
        cols = eval(f'{join_order[0].split(" ")[0]}_usecols')
        if len(cols) <= 4:
            num_col = random.randint(1, len(cols))
        else:
            num_col = random.randint(1, 5)  # number of columns

    questr = 'SELECT COUNT(*) FROM '
    questr = questr + ",".join(join_order) + " WHERE "

    if len(joins) != 0:
        for jk in joins:
            questr = questr + jk + ' AND '

    col = list(choice(cols, num_col, replace=False))

    component = []

    for k in range(len(col)):
        op = choice(ops)
        sql = f'SELECT {col[k]} FROM {dictalias[col[k].split("_")[0]]} order by random() limit 1;'
        cur = conn.cursor()
        conn.autocommit = True
        cur.execute(sql)
        val = cur.fetchall()[0][0]
        questr = questr + col[k].split('_')[0] + '.' + str(col[k]) + op + str(
            val) + ' AND '
        component.append(col[k].split('_')[0] + '.' + str(col[k]))
        component.append(op)
        component.append(val)
    questr = questr[:len(questr) - 5]
    questr += ';'
    cur.execute(questr)
    card = cur.fetchall()[0][0]
    questr += f',{card}\n'
    f.write(",".join(join_order) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
    f_sql.write(questr)
f.close()
f_sql.close()

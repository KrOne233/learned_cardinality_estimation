import csv
import numpy as np
from tqdm import tqdm
import pandas as pd
import psycopg2


def print_qerror(preds, labels):
    qerror = []
    for i in range(len(preds)):
        if preds[i] == 0:
            preds[i] = 1
        if float(labels[i]) == 0:
            labels[i] = 1
        if preds[i] > float(labels[i]):
            qerror.append(preds[i] / float(labels[i]))
        else:
            qerror.append(float(labels[i]) / float(preds[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))

    return qerror


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
        # df = pd.read_sql(sql, conn)
        # df.columns = [c.lower() for c in df.columns]
        # df[f"bit_id_{t.split(' ')[1]}"] = [i for i in range(len(df))]
        df_samples[t.split(' ')[1]] = pd.read_sql(sql, conn)

    return df_samples


def join_order(joins):
    join_tables = []
    for join in joins:
        join_tables.append(join.split('=')[0].split('.')[0])
        join_tables.append(join.split('=')[1].split('.')[0])
    rank = pd.value_counts(join_tables)
    point = []
    j = []
    for join in joins:
        p = rank[join.split('=')[0].split('.')[0]] + rank[join.split('=')[1].split('.')[0]]
        point.append(p)
        j.append(join)

    ordered_joins = list(pd.Series(point, index=j).sort_values(ascending=False).index)

    return ordered_joins


def query_on_sample(df_samples, sql_csv_line, tables):
    n_rows = 1
    for t in tables:
        n_rows = n_rows * len(df_samples[t.split(' ')[1]])

    joins = sql_csv_line[1].split(',')
    predicates = sql_csv_line[2].split(',')

    df_sample = {}

    if len(predicates[0]) > 0:
        i = 0
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
                break
            i = i + 3

    if len(tables) == 1:
        selectivity = len(list(df_sample.values())[0]) / n_rows

    else:
        for j in joins:
            t = j.split('=')[0].split('.')[0]
            if t not in df_sample:
                df_sample[t] = df_samples[t]
            t = j.split('=')[1].split('.')[0]
            if t not in df_sample:
                df_sample[t] = df_samples[t]

        joins = join_order(joins)
        j = joins[0]
        t_left = j.split('=')[0].split('.')[0]
        col_left = j.split('=')[0].split('.')[1]
        t_right = j.split('=')[1].split('.')[0]
        col_right = j.split('=')[1].split('.')[1]
        # for imdb dataset, tpch has all column names different-------------------------------------------------
        if t_left != 't':
            if 'id' in df_sample[t_left].columns:
                df_sample[t_left].drop(columns='id', inplace=True)
        if t_right != 't':
            if 'id' in df_sample[t_right].columns:
                df_sample[t_right].drop(columns='id', inplace=True)
        # ------------------------------------------------------------------------------------------------------
        if col_right == col_left:
            df_join = pd.merge(df_sample[t_left], df_sample[t_right], on=col_left)
        else:
            df_join = pd.merge(df_sample[t_left], df_sample[t_right], left_on=col_left, right_on=col_right)

        df_sample.pop(t_left)
        df_sample.pop(t_right)

        if len(joins) > 1 and len(df_join) > 0:
            for j in joins[1:]:
                t_left = j.split('=')[0].split('.')[0]
                col_left = j.split('=')[0].split('.')[1]
                t_right = j.split('=')[1].split('.')[0]
                col_right = j.split('=')[1].split('.')[1]
                try:
                    df = df_sample[t_left]
                    t = t_left
                    col = col_left
                    col_join = col_right
                except Exception:
                    df = df_sample[t_right]
                    t = t_right
                    col = col_right
                    col_join = col_left
                # for imdb dataset, tpch has all column names different-------------------------------------------------
                if t != 't':
                    if 'id' in df.columns:
                        df.drop(columns='id', inplace=True)
                # ------------------------------------------------------------------------------------------------------
                if col == col_join:
                    df_join = df_join.merge(df, on=col)
                else:
                    df_join = df_join.merge(df, left_on=col_join, right_on=col)

                df_sample.pop(t)

                if len(df_join) == 0:
                    break
        selectivity = len(df_join) / n_rows

    return selectivity


# generate samples in pandas, slower than postgre materialized view
def sample_predict(sql_csv_file, conn, samples_frac):
    cur = conn.cursor()
    conn.autocommit = True
    all_tables = load_sql_csv(sql_csv_file)
    table_names = get_all_table_names(all_tables)
    t_rows = {}
    for t in table_names:
        sql = f'SELECT COUNT(*) FROM {t}'
        cur.execute(sql)
        t_rows[t] = cur.fetchall()[0][0]
        if t_rows[t] <= 10000:
            num_samples = t_rows[t]
        else:
            # num_samples = 10000
            num_samples = int(t_rows[t] * samples_frac)
    df_samples = select_sample(table_names, conn, num_samples)
    predict = []
    true_card = []
    with open(sql_csv_file, 'r') as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for line in tqdm(lines):
            tables = line[0].split(',')
            n_rows = 1
            for t in tables:
                n_rows = n_rows * t_rows[t]
            sel = query_on_sample(df_samples, line, tables)
            predict.append(sel * n_rows)
            true_card.append(line[3])
    f.close()
    return predict, true_card


conn = psycopg2.connect(database='tpch',
                        user='postgres', password='',
                        host='localhost', port='5432'
                        )
cur = conn.cursor()
conn.autocommit = True
sql_csv_file = 'data/tpch/tpch_sql_test.csv'
predict, true_card = sample_predict(sql_csv_file, conn, 0.01)
q = print_qerror(predict, true_card)
with open('results/predicitons_sampling_imdb_db2.csv', 'w') as f:
    f.write('prediction,true\n')
    for i in range(len(predict)):
        f.write(f'{predict[i]},{true_card[i]}\n')
f.close()


# make use of matrilized view to achieve sampling
def database_sample_predict(conn, sql_csv_file, samples_frac):
    cur = conn.cursor()
    conn.autocommit = True
    all_tables = load_sql_csv(sql_csv_file)
    table_names = get_all_table_names(all_tables)
    t_rows = {}
    sample_rows = {}
    for t in table_names:
        sql = f'SELECT COUNT(*) FROM {t}'
        cur.execute(sql)
        t_rows[t] = cur.fetchall()[0][0]
        if t_rows[t] <= 10000:
            num_samples = t_rows[t]
        else:
            num_samples = int(t_rows[t] * samples_frac)
        sql = f'CREATE MATERIALIZED VIEW {t.split(" ")[1]}_sample AS SELECT * FROM {t} order by random() limit {num_samples};'
        try:
            cur.execute(sql)
        except Exception:
            cur.execute(f'DROP MATERIALIZED VIEW {t.split(" ")[1]}_sample')
            cur.execute(sql)
            print('the view is rebuilt')
        sql = f'SELECT COUNT(*) FROM {t.split(" ")[1]}_sample;'
        cur.execute(sql)
        sample_rows[t] = cur.fetchall()[0][0]

    predict = []
    true_card = []

    with open(sql_csv_file, 'r') as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for line in tqdm(lines):
            tables = line[0].split(',')
            n_rows = 1
            n_sample = 1
            view = []
            for t in tables:
                n_rows = n_rows * t_rows[t] # ture row
                n_sample = n_sample * sample_rows[t] # sample row
                view.append(f'{t.split(" ")[1]}_sample {t.split(" ")[1]}')
            if len(tables) > 1:

                sql = f'SELECT COUNT(*) FROM {",".join(view)} WHERE {" AND ".join(line[1].split(","))} AND '
            else:
                sql = f'SELECT COUNT(*) FROM {",".join(view)} WHERE '
            predicates = line[2].split(',')

            if len(predicates[0]) > 0:
                i = 0
                while i < len(predicates):
                    sql = sql + predicates[i] + predicates[i + 1] + str(predicates[i + 2]) + ' AND '
                    i = i + 3

            sql = sql[:-5] + ';'
            cur.execute(sql)
            result = cur.fetchall()[0][0]
            sel = result / n_sample
            predict.append(sel * n_rows)
            true_card.append(line[3])
    return predict, true_card


conn = psycopg2.connect(database='imdb',
                        user='postgres', password='',
                        host='localhost', port='5432'
                        )
sql_csv_file = 'data/imdb/imdb_test_sql.csv'
predict, true_card = database_sample_predict(conn, sql_csv_file, 0.01)
q = print_qerror(predict, true_card)



with open('results/predicitons_sampling_imdb_db2.csv', 'w') as f:
    f.write('prediction,true\n')
    for i in range(len(predict)):
        f.write(f'{predict[i]},{true_card[i]}\n')
f.close()

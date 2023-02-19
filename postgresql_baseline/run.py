import psycopg2
import csv
import numpy as np
from tqdm import tqdm

def gain_plan_postgre(sql_file, conn):
    conn.autocommit = True
    cursor = conn.cursor()
    prediction = []
    true_card = []
    with open(sql_file) as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for line in tqdm(lines):
            sql = 'SELECT * FROM '
            tables = line[0]
            sql = f'{sql}{tables} WHERE '
            joins = line[1].split(',')
            if len(joins[0]) > 0:      # join exist
                for i in range(len(joins)):
                    join = joins[i]
                    sql += f'{join} AND '
            predicates = line[2].split(',')
            i = 0
            while i < len(predicates) and len(predicates[0]) > 0:
                predicate = str(predicates[i])+str(predicates[i+1])+str(predicates[i+2])
                sql += f'{predicate} AND '
                i = i+3
            sql = f'{sql[:len(sql)-5]};'
            cursor.execute(cursor.mogrify('explain ' + sql))
            analyze_fetched = cursor.fetchall()
            result = str(analyze_fetched[0])
            p = result.find('rows=')
            card = int(result[p+5:].split(' ')[0])
            prediction.append(card)
            true_card.append(line[3])
    f.close()
    return prediction, true_card

# sql = '''SELECT histogram_bounds FROM pg_stats WHERE tablename='black_friday_purchase' AND attname='purchase';'''
# sql = '''SELECT null_frac, n_distinct, most_common_vals, most_common_freqs FROM pg_stats WHERE tablename='black_friday_purchase' AND attname='gender';'''

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

# for black friday purchase
conn = psycopg2.connect(database="black_friday_purchase",
                        user='postgres', password='',
                        host='localhost', port='5432'
)

sql_file = 'data/Black_Friday/black_friday_purchase_sql_test.csv'

preds, labels = gain_plan_postgre(sql_file, conn)
print_qerror(preds, labels)
with open('results/predicitons_postgre_bfp','w') as f:
    f.write('prediction,true\n')
    for i in range(len(preds)):
        f.write(f'{preds[i]},{labels[i]}\n')
f.close()

#for fps
conn = psycopg2.connect(database="fps",
                        user='postgres', password='',
                        host='localhost', port='5432'
)

sql_file = 'data/gpu_game_fps/fps_sql_test.csv'
preds, labels = gain_plan_postgre(sql_file, conn)
print_qerror(preds, labels)
with open('results/predicitons_postgre_fps','w') as f:
    f.write('prediction,true\n')
    for i in range(len(preds)):
        f.write(f'{preds[i]},{labels[i]}\n')
f.close()


# for bfp 2_col_i
conn = psycopg2.connect(database="black_friday_purchase",
                        user='postgres', password='',
                        host='localhost', port='5432'
)
i=1
while i<8:
    sql_file = f'data/Black_Friday/2_col_sql/bfp_2col_{i}_test.csv'
    preds, labels = gain_plan_postgre(sql_file, conn)
    print_qerror(preds, labels)
    with open(f'results/predicitons_postgre_bfp_2col_{i}.csv', 'w') as f:
        f.write('prediction,true\n')
        for v in range(len(preds)):
            f.write(f'{preds[v]},{labels[v]}\n')
    f.close()
    i=i+2


#for fps 2_col_i
conn = psycopg2.connect(database="fps",
                        user='postgres', password='',
                        host='localhost', port='5432'
)
i=1
while i<38:
    sql_file = f'data/gpu_game_fps/2_col_sql/fps_2col_{i}_test.csv'
    preds, labels = gain_plan_postgre(sql_file, conn)
    print_qerror(preds, labels)
    with open(f'results/predicitons_postgre_fps_2col_{i}.csv', 'w') as f:
        f.write('prediction,true\n')
        for v in range(len(preds)):
            f.write(f'{preds[v]},{labels[v]}\n')
    f.close()
    i=i+4

# imdb
conn = psycopg2.connect(database="imdb",
                        user='postgres', password='',
                        host='localhost', port='5432'
)

sql_file = 'data/imdb/imdb_test_sql.csv'
preds, labels = gain_plan_postgre(sql_file, conn)
print_qerror(preds, labels)
with open('results/predicitons_postgre_imdb.csv','w') as f:
    f.write('prediction,true\n')
    for i in range(len(preds)):
        f.write(f'{preds[i]},{labels[i]}\n')
f.close()


# tpch
conn = psycopg2.connect(database="tpch",
                        user='postgres', password='',
                        host='localhost', port='5432'
)

sql_file = 'data/tpch/tpch_sql_test.csv'
preds, labels = gain_plan_postgre(sql_file, conn)
print_qerror(preds, labels)
with open('results/predicitons_postgre_tpch.csv','w') as f:
    f.write('prediction,true\n')
    for i in range(len(preds)):
        f.write(f'{preds[i]},{labels[i]}\n')
f.close()

# tpch_multi_joins
conn = psycopg2.connect(database="tpch",
                        user='postgres', password='',
                        host='localhost', port='5432'
)
for j in range(0,7):
    sql_file = f'data/tpch/more_joins/tpch_sql_test_{j}.csv'
    preds, labels = gain_plan_postgre(sql_file, conn)
    print_qerror(preds, labels)
    with open(f'results/predicitons_postgre_tpch_{j}.csv', 'w') as f:
        f.write('prediction,true\n')
        for i in range(len(preds)):
            f.write(f'{preds[i]},{labels[i]}\n')
    f.close()

# imdb multi joins
conn = psycopg2.connect(database="imdb",
                        user='postgres', password='',
                        host='localhost', port='5432'
)
for j in range(0,5):
    sql_file = f'data/imdb/more_joins/imdb_sql_test_{j}.csv'
    preds, labels = gain_plan_postgre(sql_file, conn)
    print_qerror(preds, labels)
    with open(f'results/predicitons_postgre_imdb_{j}.csv', 'w') as f:
        f.write('prediction,true\n')
        for i in range(len(preds)):
            f.write(f'{preds[i]},{labels[i]}\n')
    f.close()
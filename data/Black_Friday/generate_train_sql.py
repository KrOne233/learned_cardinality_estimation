import random
from numpy.random import choice
import pandas as pd
import psycopg2
from tqdm import tqdm

df_purchase = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv',sep=',', escapechar='\\', encoding='utf-8',
                       low_memory=False, quotechar='"')

conn = psycopg2.connect(database="Master_thesis",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
)
conn.autocommit = True
cursor = conn.cursor()


dictalias = {'black_friday_purchase': ['bfp']}
t_col = list(df_purchase.columns)[1:] # id is not used for filtering
ops = ['=', '<', '>'] # operations
predicates = []
tables = ['black_friday_purchase bfp']
joins = []
f = open("data/Black_Friday/black_friday_purchase_sql.csv", 'w')
f_sql = open("data/Black_Friday/black_friday_purchase.sql", 'w')
for i in tqdm(range(40000)):
    questr = 'SELECT * FROM '
    questr = questr + ",".join(tables) + " WHERE "
    num_col = random.randint(1, len(t_col)) # number of columns
    col = list(choice(t_col, num_col, replace=False))
    component = []
    result = []
    for k in range(num_col):
        component.append(dictalias['black_friday_purchase'][0] + '.' + str(col[k]))
        op = choice(ops)
        component.append(op)
        val = int(df_purchase[col[k]][random.randint(0, len(df_purchase[col[k]])-1)])
        component.append(val)
        questr += dictalias['black_friday_purchase'][0] + '.' + str(col[k]) + op + str(val) + ' AND ' # need to change if join exist
    questr = questr[:len(questr) - 5]
    questr += ';'
    # df = pd.read_sql(questr, conn)
    # card = df['count'].values[0]
    cursor.execute(questr)
    card = len(cursor.fetchall())
    questr += f',{card}\n'
    predicates.append(component)
    f.write(",".join(tables) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
    f_sql.write(questr)
f.close()
f_sql.close()

# 生成所有属性组合的sql

# generate train and testset
with open("data/Black_Friday/black_friday_purchase_sql.csv", "r") as input:
    lines = input.readlines()
    with open("data/Black_Friday/black_friday_purchase_sql_train.csv", "w") as output_train:
        with open("data/Black_Friday/black_friday_purchase_sql_test.csv", "w") as output_test:
            i = 0
            testlines = random.sample(range(len(lines)), 1000)
            for line in lines:
                if i in testlines:
                    output_test.write(line)
                else:
                    output_train.write(line)
                i = i+1



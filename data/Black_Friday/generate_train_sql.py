import os
import random
from numpy.random import choice
import pandas as pd
import psycopg2
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.formula.api import ols

df_purchase = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv', sep=',', escapechar='\\', encoding='utf-8',
                          low_memory=False, quotechar='"')

conn = psycopg2.connect(database="Master_thesis",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
                        )
conn.autocommit = True
cursor = conn.cursor()

dictalias = {'black_friday_purchase': ['bfp']}
t_col = list(df_purchase.columns)[1:]  # id is not used for filtering
ops = ['=', '<', '>']  # operations
predicates = []
tables = ['black_friday_purchase bfp']
joins = []
f = open("data/Black_Friday/black_friday_purchase_sql.csv", 'w')
f_sql = open("data/Black_Friday/black_friday_purchase.sql", 'w')
for i in tqdm(range(40000)):
    questr = 'SELECT COUNT(*) FROM '
    questr = questr + ",".join(tables) + " WHERE "
    num_col = random.randint(1, len(t_col))  # number of columns
    col = list(choice(t_col, num_col, replace=False))
    component = []
    result = []
    for k in range(num_col):
        component.append(dictalias['black_friday_purchase'][0] + '.' + str(col[k]))
        op = choice(ops)
        component.append(op)
        val = int(df_purchase[col[k]][random.randint(0, len(df_purchase[col[k]]) - 1)])
        component.append(val)
        questr += dictalias['black_friday_purchase'][0] + '.' + str(col[k]) + op + str(
            val) + ' AND '  # need to change if join exist
    questr = questr[:len(questr) - 5]
    questr += ';'
    # df = pd.read_sql(questr, conn)
    # card = df['count'].values[0]
    cursor.execute(questr)
    card = cursor.fetchall()[0][0]
    questr += f',{card}\n'
    predicates.append(component)
    f.write(",".join(tables) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
    f_sql.write(questr)
f.close()
f_sql.close()


# evaluate correlations between attributes to final purchase
df_purchase = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv')
df_purchase.columns = [c.lower() for c in df_purchase.columns]
model = ols('purchase ~ gender + age + occupation + city_category + stay_in_current_city_years + marital_status + '
            'product_category_1 + product_category_2 + product_category_3', data=df_purchase).fit()
anova = sm.stats.anova_lm(model, typ=2)


# generate N-columns correlation query set
''' fast but lot of 0
def N_col_sql(num_col, sorted_colset, order, f, f_sql, df, tables, dictalias, conn):
    conn.autocommit = True
    cursor = conn.cursor()
    ops = ['=', '<', '>']
    predicates = []
    joins = []
    for i in tqdm(range(40000)):
        questr = 'SELECT COUNT(*) FROM '
        questr = questr + ",".join(tables) + " WHERE "
        col = sorted_colset[order - 1:order + num_col - 2]
        col.append('Purchase')
        component = []
        result = []
        count = 0
        for k in range(len(col)):
            component.append(dictalias['black_friday_purchase'][0] + '.' + str(col[k]))
            op = choice(ops)
            component.append(op)
            val = int(df[col[k]][random.randint(0, len(df[col[k]]) - 1)])
            component.append(val)
            questr += dictalias['black_friday_purchase'][0] + '.' + str(col[k]) + op + str(
                val) + ' AND '  # need to change if join exist
        questr = questr[:len(questr) - 5]
        questr += ';'
        # df_temp = pd.read_sql(questr, conn)
        # card = df['count'].values[0]
        cursor.execute(questr)
        card = cursor.fetchall()[0][0]
        questr += f',{card}\n'
        predicates.append(component)
        f.write(",".join(tables) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
        f_sql.write(questr)
    f.close()
    f_sql.close()
'''
# 每次选好一列的filter条件，就进行查询，返回结果，下一列的值在这个结果中选，同时如果出现查询结果为空，尝试10次别的值. slow but few 0
def N_col_sql(num_col, sorted_colset, order, f, f_sql, df, tables, dictalias, conn):
    df.columns = [c.lower() for c in df.columns]
    conn.autocommit = True
    cursor = conn.cursor()
    ops = ['=', '<', '>']
    predicates = []
    joins = []
    col = sorted_colset[order - 1:num_col + order - 2]
    col.append('purchase')

    for i in tqdm(range(10000)):
        questr = 'SELECT COUNT(*) FROM '
        questr = questr + ",".join(tables) + " WHERE "
        component = []
        result = []
        df_temp = df
        for k in range(len(col)):
            component.append(dictalias['black_friday_purchase'][0] + '.' + str(col[k]))
            op = choice(ops)
            component.append(op)
            val = choice(list(set(df_temp[col[k]])))
            component.append(val)
            questr_temp = questr + dictalias['black_friday_purchase'][0] + '.' + str(col[k]) + op + str(
                val) + ' AND '
            questr_temp = questr_temp[:len(questr_temp) - 5].replace('COUNT(*)', ','.join(col)) # only select required columns, reduce I/O
            df_temp_pre = df_temp  # 保存上次循环的df_temp, 给后面10次重新选择用，否则在df_temp为空的情况下只能df_temp = df
            df_temp = pd.read_sql(questr_temp, conn)
            #if len(df_temp) == 0:
            #    df_temp = df
            count = 0
            while (len(df_temp) == 0):
                # df_temp = df
                component = component[:len(component) - 2]
                op = choice(ops)
                component.append(op)
                val = choice(list(set(df_temp_pre[col[k]])))
                component.append(val)
                questr_temp = questr + dictalias['black_friday_purchase'][0] + '.' + str(col[k]) + op + str(
                    val) + ' AND '
                questr_temp = questr_temp[:len(questr_temp) - 5].replace('COUNT(*)', ','.join(col))
                df_temp = pd.read_sql(questr_temp, conn)
                count = count + 1
                if count > 5:
                    if len(df_temp) == 0:
                        df_temp = df
                    break
            questr = questr + dictalias['black_friday_purchase'][0] + '.' + str(col[k]) + op + str(
                val) + ' AND '
        questr = questr[:len(questr) - 5]
        questr += ';'
        # df_temp = pd.read_sql(questr, conn)
        # card = df['count'].values[0]
        cursor.execute(questr)
        card = cursor.fetchall()[0][0]
        questr += f',{card}\n'
        predicates.append(component)
        f.write(",".join(tables) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
        f_sql.write(questr)
    f.close()
    f_sql.close()

sorted_colset = list(anova.iloc[:, -1].sort_values().index)
conn = psycopg2.connect(database="black_friday_purchase",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432')

dictalias = {'black_friday_purchase': ['bfp']}
tables = ['black_friday_purchase bfp']

# 2 cols
os.makedirs('data/Black_Friday/2_col_sql', exist_ok=True)
for i in range(1, 8):
    f = open(f"data/Black_Friday/2_col_sql/bfp_2col_sql_{i}.csv", 'w')
    f_sql = open(f"data/Black_Friday/2_col_sql/bfp_2col_sql_{i}.sql", 'w')
    N_col_sql(2, sorted_colset, i, f, f_sql, df_purchase, tables, dictalias, conn)

# 3 cols
os.makedirs('data/Black_Friday/3_col_sql', exist_ok=True)
f = open("data/Black_Friday/3_col_sql/bfp_3col_sql_1.csv", 'w')
f_sql = open("data/Black_Friday/3_col_sql/bfp_3col_sql_1.sql", 'w')
N_col_sql(3, sorted_colset, 1, f, f_sql, df_purchase, tables, dictalias, conn)



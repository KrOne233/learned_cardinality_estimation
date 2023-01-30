import os
import random
from numpy.random import choice
import pandas as pd
import psycopg2
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.formula.api import ols
def N_col_sql(num_col, sorted_colset, order, f, f_sql, df, tables, dictalias, conn):
    df.columns = [c.lower() for c in df.columns]
    conn.autocommit = True
    cursor = conn.cursor()
    ops = ['=', '<', '>']
    joins = []
    col = sorted_colset[order - 1:num_col + order - 2]
    col.append('fps')

    for i in tqdm(range(10000)):
        questr = 'SELECT COUNT(*) FROM '
        questr = questr + ",".join(tables) + " WHERE "
        component = []
        result = []
        df_temp = df
        for k in range(len(col)):
            op = choice(ops)
            dist_val_list = list(set(df_temp[col[k]]))
            val = choice(dist_val_list)
            questr_temp = questr + dictalias['fps'][0] + '.' + str(col[k]) + op + str(val)
            if k < len(col) - 1:
                questr_temp = questr_temp.replace('COUNT(*)', col[k + 1])  # only select required columns, reduce I/O
            else:
                questr_temp = questr_temp.replace('COUNT(*)', col[k])
            df_temp = pd.read_sql(questr_temp, conn)
            # if len(df_temp) == 0:
            #    df_temp = df
            count = 0
            while (len(df_temp) == 0):
                # df_temp = df
                op = choice(ops)
                val = choice(dist_val_list)
                questr_temp = questr + dictalias['fps'][0] + '.' + str(col[k]) + op + str(val)
                if k < len(col) - 1:
                    questr_temp = questr_temp.replace('COUNT(*)',
                                                      col[k + 1])  # only select required columns, reduce I/O
                else:
                    questr_temp = questr_temp.replace('COUNT(*)', col[k])
                df_temp = pd.read_sql(questr_temp, conn)
                count = count + 1
                if count > 2:
                    if len(df_temp) == 0:
                        df_temp = df
                    break
            questr = questr + dictalias['fps'][0] + '.' + str(col[k]) + op + str(
                val) + ' AND '
            component.append(dictalias['fps'][0] + '.' + str(col[k]))
            component.append(op)
            component.append(val)
        questr = questr[:len(questr) - 5]
        questr += ';'
        # df_temp = pd.read_sql(questr, conn)
        # card = df['count'].values[0]
        cursor.execute(questr)
        card = cursor.fetchall()[0][0]
        questr += f',{card}\n'
        f.write(",".join(tables) + '#' + ','.join(joins) + '#' + ",".join(map(str, component)) + '#' + str(card) + '\n')
        f_sql.write(questr)
    f.close()
    f_sql.close()



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

model = ols('fps ~ cpunumberofcores + cpunumberofthreads + cpubaseclock + cpucachel1 + cpucachel2 + cpucachel3\
                    + cpudiesize+ cpufrequency + cpumultiplier + cpumultiplierunlocked + cpuprocesssize + cputdp\
                    + cpunumberoftransistors + cputurboclock\
                    + gpubandwidth + gpubaseclock + gpuboostclock + gpubusnterface\
                    + gpunumberofcomputeunits + gpudiesize + gpunumberofexecutionunits\
                    + gpufp32performance + gpumemorybus + gpumemorysize + gpumemorytype + gpupixelrate\
                    + gpuprocesssize + gpunumberofrops + gpushadermodel + gpunumberofshadingunits\
                    + gpunumberoftmus + gputexturerate + gpunumberoftransistors\
                    + gpuvulkan + gamename + gameresolution + gamesetting', data=df_fps).fit()

anova = sm.stats.anova_lm(model, typ=2)
sorted_colset = list(anova.iloc[:, -1].sort_values().index)
sorted_colset = sorted_colset[:-1]

dictalias = {'fps': ['f']}
tables = ['fps f']

conn = psycopg2.connect(database="fps",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
                        )

# 2 cols
i=5
os.makedirs('data/gpu_game_fps/2_col_sql', exist_ok=True)
while i <= len(sorted_colset):
    f = open(f"data/gpu_game_fps/2_col_sql/fps_2col_sql_{i}.csv", 'w')
    f_sql = open(f"data/gpu_game_fps/2_col_sql/fps_2col_sql_{i}.sql", 'w')
    N_col_sql(2, sorted_colset, i, f, f_sql, df_fps, tables, dictalias, conn)
    i = i+8

def gen_train_test(path_sql_file, path_train_file, path_test_file, num_test):
    with open(path_sql_file, "r") as input:
        lines = input.readlines()
        with open(path_train_file, "w") as output_train:
            with open(path_test_file, "w") as output_test:
                i = 0
                testlines = random.sample(range(len(lines)), num_test)
                for line in lines:
                    if i in testlines:
                        output_test.write(line)
                    else:
                        output_train.write(line)
                    i = i + 1
            output_test.close()
        output_train.close()
    input.close()

i = 1
while i<38:
    path_sql_file = f"data/gpu_game_fps/2_col_sql/fps_2col_sql_{i}.csv"
    path_train_file = f'data/gpu_game_fps/2_col_sql/fps_2col_{i}_train.csv'
    path_test_file = f'data/gpu_game_fps/2_col_sql/fps_2col_{i}_test.csv'
    num_test = 500
    gen_train_test(path_sql_file, path_train_file, path_test_file, num_test)
    i = i+4


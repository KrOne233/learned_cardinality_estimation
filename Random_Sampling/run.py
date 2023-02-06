import psycopg2
import csv
import numpy as np
import pandas as pd


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


def query_on_sample(df_sample, sql_csv_line):
    n_rows = len(df_sample)
    predicates = sql_csv_line[2].split(',')
    i = 0
    while i < len(predicates):
        predicate = str(predicates[i])+str(predicates[i+1])+str(predicates[i+2])
        if predicates[i+1] == '=':
            predicates[i + 1] = '=='
        df_sample = df_sample[eval('df_sample[predicates[i].split(".")[1]]'+predicates[i+1]+predicates[i+2])]
        if len(df_sample)==0:
            break
        i = i+3
    selectivity = len(df_sample)/n_rows
    return selectivity


def sample_predict(sql_csv_file, df_sample, min_max_file):  # single table
    with open(min_max_file) as m:
        n_rows = int(m.readlines()[1].split(',')[3])
    m.close()
    predict = []
    true_card = []
    with open(sql_csv_file) as f:
        lines = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for line in lines:
            sel = query_on_sample(df_sample, line)
            predict.append(sel*n_rows)
            true_card.append(line[3])
    f.close()
    return predict, true_card




conn = psycopg2.connect(database="black_friday_purchase",
                        user='postgres', password='wzy07wx25',
                        host='localhost', port='5432'
)

# for black friday purchase
black_friday_purchase = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv', sep=',', escapechar='\\', encoding='utf-8',
                          low_memory=False, quotechar='"').sample(5000)
black_friday_purchase.columns = [c.lower() for c in black_friday_purchase.columns]
sql_file = 'data/Black_Friday/black_friday_purchase_sql_test.csv'
min_max_file = 'data/Black_Friday/column_min_max_vals.csv'

predict, true_card = sample_predict(sql_file,black_friday_purchase,min_max_file)
print_qerror(predict,true_card)
with open('results/predicitons_sampling_bfp.csv','w') as f:
    f.write('prediction,true\n')
    for i in range(len(predict)):
        f.write(f'{predict[i]},{true_card[i]}\n')
f.close()



# for fps
fps = pd.read_csv('data/gpu_game_fps/fps_num_lower.csv', sep=',', escapechar='\\', encoding='utf-8',
                          low_memory=False, quotechar='"').sample(5000)
sql_file = 'data/gpu_game_fps/fps_sql_test.csv'
min_max_file = 'data/gpu_game_fps/column_min_max_vals.csv'
predict, true_card = sample_predict(sql_file,fps,min_max_file)
print_qerror(predict,true_card)
with open('results/predicitons_sampling_fps.csv','w') as f:
    f.write('prediction,true\n')
    for i in range(len(predict)):
        f.write(f'{predict[i]},{true_card[i]}\n')
f.close()

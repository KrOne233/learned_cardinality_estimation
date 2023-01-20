import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
import MultiHist.multihist as multi_hist


# Functions
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_query(file_name):
    joins = []
    predicates = []
    tables = []
    label = []

    # Load queries
    with open(file_name + ".csv", 'rU') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))
            joins.append(row[1].split(','))
            predicates.append(row[2].split(','))
            if int(row[3]) < 1:
                label.append(1)
                # print("Queries must have non-zero cardinalities")
                # exit(1)
            else:
                label.append(row[3])
    print("Loaded queries")
    predicates = [list(chunks(d, 3)) for d in predicates]
    return joins, predicates, tables, label


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] == 0:
            preds_unnorm[i] = 1

        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


# predictions
def predict(data_file, table_name, usecols, sql_csv_name, max_partition):
    table = multi_hist.LoadMyDataset(data_file,
                                     table_name, usecols)
    estimator = multi_hist.MaxDiffHistogram(table, max_partition)

    joins, predicates, tables, label = load_query(sql_csv_name)

    col_dic = {}
    for col in table.columns:
        col_dic[col.Name()] = col

    predicts = []
    for query in tqdm(predicates):
        for p in query:
            ops = []
            columns = []
            vals = []
            if len(p) == 3:
                column_name = p[0].split('.')[-1]
                columns.append(col_dic[column_name])
                ops.append(p[1])
                if table.data[column_name].dtype == 'int64':
                    vals.append(int(p[2]))
                elif table.data[column_name].dtype == 'float64':
                    vals.append(float(p[2]))
        predicts.append(estimator.Query(columns, ops, vals))

    print_qerror(predicts, label)

    results = pd.DataFrame({
        'predicts': predicts,
        'label': label
    })
    results.to_csv(f'results/predictions_MutiHist_{table_name}.csv', index=False)


data_file = 'data/gpu_game_fps/fps_num_lower.csv'
table_name = 'fps'
sql_csv_name = 'data/gpu_game_fps/fps_sql_test'
usecols = ['cpunumberofcores', 'cpunumberofthreads',
           'cpubaseclock', 'cpucachel1', 'cpucachel2', 'cpucachel3', 'cpudiesize',
           'cpufrequency', 'cpumultiplier', 'cpumultiplierunlocked',
           'cpuprocesssize', 'cputdp', 'cpunumberoftransistors', 'cputurboclock',
           'gpubandwidth', 'gpubaseclock', 'gpuboostclock', 'gpubusnterface',
           'gpunumberofcomputeunits', 'gpudiesize', 'gpunumberofexecutionunits',
           'gpufp32performance', 'gpumemorybus', 'gpumemorysize', 'gpumemorytype', 'gpupixelrate',
           'gpuprocesssize', 'gpunumberofrops', 'gpushadermodel', 'gpunumberofshadingunits',
           'gpunumberoftmus', 'gputexturerate', 'gpunumberoftransistors',
           'gpuvulkan', 'gamename', 'gameresolution', 'gamesetting', 'fps']
predict(data_file, table_name, usecols, sql_csv_name, 1000)

# test
import MultiHist.multihist_vf as multi_hist_vf
test = pd.read_csv('MultiHist/test.csv')
test = test.astype('float64')
test.to_csv('MultiHist/test_float.csv',index=False)

table = multi_hist.LoadMyDataset('MultiHist/test_float.csv',
                                 'test', usecols=['A', 'B'])
table = multi_hist.LoadMyDataset('MultiHist/test.csv',
                                 'test', usecols=['A', 'B'])
estimator = multi_hist.MaxDiffHistogram(table, 4)
estimator = multi_hist_vf.MaxDiffHistogram(table,4)
estimator.Query(table.columns, ['<','>'], [2.1, 1.1])

# black_friday_purchase
data_file = 'data/Black_Friday/Black_Friday_Purchase_num.csv'
table_name = 'black_friday_purchase'
sql_csv_name = 'data/Black_Friday/black_friday_purchase_sql_test'
usecols = ['Gender', 'Age', 'Occupation', 'City_Category','Stay_In_Current_City_Years', 'Marital_Status',
           'Product_Category_1','Product_Category_2','Product_Category_3', 'Purchase']
predict(data_file, table_name, usecols, sql_csv_name, 30000)

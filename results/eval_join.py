import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from scipy import stats

pio.renderers.default = "browser"


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if np.isnan(preds_unnorm[i]):
            preds_unnorm[i] = 1
        if preds_unnorm[i] < 1:
            preds_unnorm[i] = 1
        if float(labels_unnorm[i]) < 1:
            labels_unnorm[i] = 1
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
    return qerror


# imdb
postgre_imdb = pd.read_csv('results/predicitons_postgre_imdb.csv')
q_postgre_imdb = print_qerror(postgre_imdb.iloc[:, 0], postgre_imdb.iloc[:, 1])
sampling_imdb = pd.read_csv('results/predicitons_sampling_imdb_resample2.csv')
q_sampling_imdb = print_qerror(sampling_imdb.iloc[:, 0], sampling_imdb.iloc[:, 1])
mscn_imdb = pd.read_csv('results/predictions_mscn_test_imdb_no_bitmap.csv', header=None)
mscn_bitmap_imdb = pd.read_csv('results/predictions_mscn_test_imdb_bitmap.csv', header=None)
q_mscn_imdb = print_qerror(mscn_imdb.iloc[:, 0], mscn_imdb.iloc[:, 1])
q_mscn_bitmap_imdb = print_qerror(mscn_bitmap_imdb.iloc[:, 0], mscn_bitmap_imdb.iloc[:, 1])
xgb_imdb = pd.read_csv('results/predictions_xgb_imdb.csv', header=None)
q_xbg_imdb = print_qerror(xgb_imdb.iloc[:, 0], xgb_imdb.iloc[:, 1])
deepdb_imdb = pd.read_csv('results/result_deepdb_imdb.csv',
                          usecols=['cardinality_predict', 'cardinality_true'])
q_deepdb_imdb = print_qerror(deepdb_imdb.iloc[:, 0], deepdb_imdb.iloc[:, 1])

# q_sampling_imdb_db = pd.read_csv('results/predicitons_sampling_imdb_db2.csv')
# q_sampling_imdb_db = print_qerror(q_sampling_imdb_db.iloc[:, 0], q_sampling_imdb_db.iloc[:, 1])

q_imdb = pd.DataFrame({'postgre': q_postgre_imdb, 'sampling': q_sampling_imdb, 'sampling_db': q_sampling_imdb_db,
                       'mscn': q_mscn_imdb, 'mscn_bitmap': q_mscn_bitmap_imdb, 'xgb': q_xbg_imdb,
                       'deepdb': q_deepdb_imdb})

q_imdb.to_csv('results/q_imdb.csv', index=False, encoding='utf-8')

# tpch
postgre_tpch = pd.read_csv('results/table joins/tpch/predicitons_postgre_tpch.csv')
q_postgre_tpch = print_qerror(postgre_tpch.iloc[:, 0], postgre_tpch.iloc[:, 1])

sampling_tpch = pd.read_csv('results/table joins/tpch/predicitons_sampling_tpch_db2.csv')
q_sampling_tpch = print_qerror(sampling_tpch.iloc[:, 0], sampling_tpch.iloc[:, 1])

mscn_tpch = pd.read_csv('results/table joins/tpch/predictions_mscn_test_tpch.csv', header=None)
mscn_bitmap_tpch = pd.read_csv('results/table joins/tpch/predictions_mscn_test_tpch_bitmap.csv', header=None)
q_mscn_tpch = print_qerror(mscn_tpch.iloc[:, 0], mscn_tpch.iloc[:, 1])
q_mscn_bitmap_tpch = print_qerror(mscn_bitmap_tpch.iloc[:, 0], mscn_bitmap_tpch.iloc[:, 1])

xgb_tpch = pd.read_csv('results/table joins/tpch/predictions_xgb_tpch.csv', header=None)
q_xbg_tpch = print_qerror(xgb_tpch.iloc[:, 0], xgb_tpch.iloc[:, 1])

deepdb_tpch = pd.read_csv('results/table joins/tpch/result_deepdb_tpch.csv',
                          usecols=['cardinality_predict', 'cardinality_true'])
preds = deepdb_tpch.iloc[:, 0]
true = deepdb_tpch.iloc[:, 1]
q_deepdb_tpch = print_qerror(preds, true)

q_tpch = pd.DataFrame({'postgre': q_postgre_tpch, 'sampling': q_sampling_tpch,
                       'mscn': q_mscn_tpch, 'mscn_bitmap': q_mscn_bitmap_tpch, 'xgb': q_xbg_tpch,
                       'deepdb': q_deepdb_tpch})
q_tpch.to_csv('results/q_tpch.csv', index=False, encoding='utf-8')

# anova test
q_tpch = pd.read_csv('results/table joins/tpch/q_tpch.csv')
q_imdb = pd.read_csv('results/table joins/imdb/q_imdb.csv')
anova_postgre = stats.f_oneway(q_imdb['postgre'], q_tpch['postgre'])
anova_sampling = stats.f_oneway(q_imdb['sampling'], q_tpch['sampling'])
anova_mscn = stats.f_oneway(q_imdb['mscn'], q_tpch['mscn'])
anova_mscn_bitmap = stats.f_oneway(q_imdb['mscn_bitmap'], q_tpch['mscn_bitmap'])
anova_xgb = stats.f_oneway(q_imdb['xgb'], q_tpch['xgb'])
anova_deepdb = stats.f_oneway(q_imdb['deepdb'], q_tpch['deepdb'])

# multi-joins
# postgre
postgre_imdb_joins = []
for i in range(0, 5):
    postgre_imdb_join = pd.read_csv(f'results/table joins/multi_joins/postgre/predicitons_postgre_imdb_{i}.csv')
    q_postgre_imdb_join = print_qerror(postgre_imdb_join.iloc[:, 0], postgre_imdb_join.iloc[:, 1])
    postgre_imdb_joins.append(q_postgre_imdb_join)

for i in range(0, 5):
    with open(f'results/table joins/multi_joins/postgre/q_postgre_imdb_{i}.csv', 'w') as f:
        for q in postgre_imdb_joins[i]:
            f.write(str(q) + '\n')

anova_postgre_imdb = stats.f_oneway(postgre_imdb_joins[0], postgre_imdb_joins[1],
                                    postgre_imdb_joins[2], postgre_imdb_joins[3], postgre_imdb_joins[4])
anova_postgre_imdb_2join = stats.f_oneway(postgre_imdb_joins[0], postgre_imdb_joins[1],
                                          postgre_imdb_joins[2])

postgre_tpch_joins = []
for i in range(0, 7):
    postgre_tpch_join = pd.read_csv(f'results/table joins/multi_joins/postgre/predicitons_postgre_tpch_{i}.csv')
    q_postgre_tpch_join = print_qerror(postgre_tpch_join.iloc[:, 0], postgre_tpch_join.iloc[:, 1])
    postgre_tpch_joins.append(q_postgre_tpch_join)

for i in range(0, 7):
    with open(f'results/table joins/multi_joins/postgre/q_postgre_tpch_{i}.csv', 'w') as f:
        for q in postgre_tpch_joins[i]:
            f.write(str(q) + '\n')

anova_postgre_tpch = stats.f_oneway(postgre_tpch_joins[0], postgre_tpch_joins[1],
                                    postgre_tpch_joins[2], postgre_tpch_joins[3], postgre_tpch_joins[4],
                                    postgre_tpch_joins[5], postgre_tpch_joins[6])

anova_postgre_tpch_2join = stats.f_oneway(postgre_tpch_joins[0], postgre_tpch_joins[1],
                                    postgre_tpch_joins[2])
# sampling
sampling_imdb_joins = []
for i in range(0, 5):
    sampling_imdb_join = pd.read_csv(f'results/table joins/multi_joins/sampling/predicitons_sampling_imdb_{i}.csv')
    q_sampling_imdb_join = print_qerror(sampling_imdb_join.iloc[:, 0], sampling_imdb_join.iloc[:, 1])
    sampling_imdb_joins.append(q_sampling_imdb_join)

for i in range(0, 5):
    with open(f'results/table joins/multi_joins/sampling/q_sampling_imdb_{i}.csv', 'w') as f:
        for q in sampling_imdb_joins[i]:
            f.write(str(q) + '\n')

anova_sampling_sampling = stats.f_oneway(sampling_imdb_joins[0], sampling_imdb_joins[1],
                                         sampling_imdb_joins[2], sampling_imdb_joins[3], sampling_imdb_joins[4])

sampling_tpch_joins = []
for i in range(0, 7):
    sampling_tpch_join = pd.read_csv(f'results/table joins/multi_joins/sampling/predicitons_sampling_tpch_{i}.csv')
    q_sampling_tpch_join = print_qerror(sampling_tpch_join.iloc[:, 0], sampling_tpch_join.iloc[:, 1])
    sampling_tpch_joins.append(q_sampling_tpch_join)

for i in range(0, 7):
    with open(f'results/table joins/multi_joins/sampling/q_sampling_tpch_{i}.csv', 'w') as f:
        for q in sampling_tpch_joins[i]:
            f.write(str(q) + '\n')

anova_sampling_tpch = stats.f_oneway(sampling_tpch_joins[0], sampling_tpch_joins[1],
                                     sampling_tpch_joins[2], sampling_tpch_joins[3], sampling_tpch_joins[4],
                                     sampling_tpch_joins[5], sampling_tpch_joins[6])

# mscn
mscn_imdb_joins = []
for i in range(0, 5):
    mscn_imdb_join = pd.read_csv(f'results/table joins/multi_joins/mscn/predictions_mscn_test_imdb_sql_test_{i}.csv')
    q_mscn_imdb_join = print_qerror(mscn_imdb_join.iloc[:, 0], mscn_imdb_join.iloc[:, 1])
    mscn_imdb_joins.append(q_mscn_imdb_join)

for i in range(0, 5):
    with open(f'results/table joins/multi_joins/mscn/q_mscn_imdb_{i}.csv', 'w') as f:
        for q in mscn_imdb_joins[i]:
            f.write(str(q) + '\n')

anova_mscn_sampling = stats.f_oneway(mscn_imdb_joins[0], mscn_imdb_joins[1],
                                     mscn_imdb_joins[2], mscn_imdb_joins[3], mscn_imdb_joins[4])

mscn_tpch_joins = []
for i in range(0, 7):
    mscn_tpch_join = pd.read_csv(f'results/table joins/multi_joins/mscn/predictions_mscn_test_tpch_sql_test_{i}.csv')
    q_mscn_tpch_join = print_qerror(mscn_tpch_join.iloc[:, 0], mscn_tpch_join.iloc[:, 1])
    mscn_tpch_joins.append(q_mscn_tpch_join)

for i in range(0, 7):
    with open(f'results/table joins/multi_joins/mscn/q_mscn_tpch_{i}.csv', 'w') as f:
        for q in mscn_tpch_joins[i]:
            f.write(str(q) + '\n')

anova_mscn_tpch = stats.f_oneway(mscn_tpch_joins[0], mscn_tpch_joins[1],
                                 mscn_tpch_joins[2], mscn_tpch_joins[3], mscn_tpch_joins[4],
                                 mscn_tpch_joins[5], mscn_tpch_joins[6])

# mscn_bitmap
mscn_bitmap_imdb_joins = []
for i in range(0, 5):
    mscn_bitmap_imdb_join = pd.read_csv(
        f'results/table joins/multi_joins/mscn_bitmap/predictions_mscn_test_imdb_sql_test_{i}_bitmap.csv')
    q_mscn_bitmap_imdb_join = print_qerror(mscn_bitmap_imdb_join.iloc[:, 0], mscn_bitmap_imdb_join.iloc[:, 1])
    mscn_bitmap_imdb_joins.append(q_mscn_bitmap_imdb_join)

for i in range(0, 5):
    with open(f'results/table joins/multi_joins/mscn_bitmap/q_mscn_bitmap_imdb_{i}.csv', 'w') as f:
        for q in mscn_bitmap_imdb_joins[i]:
            f.write(str(q) + '\n')

anova_mscn_bitmap_sampling = stats.f_oneway(mscn_bitmap_imdb_joins[0], mscn_bitmap_imdb_joins[1],
                                            mscn_bitmap_imdb_joins[2], mscn_bitmap_imdb_joins[3],
                                            mscn_bitmap_imdb_joins[4])

mscn_bitmap_tpch_joins = []
for i in range(0, 7):
    mscn_bitmap_tpch_join = pd.read_csv(
        f'results/table joins/multi_joins/mscn_bitmap/predictions_mscn_test_tpch_sql_test_{i}_bitmap.csv')
    q_mscn_bitmap_tpch_join = print_qerror(mscn_bitmap_tpch_join.iloc[:, 0], mscn_bitmap_tpch_join.iloc[:, 1])
    mscn_bitmap_tpch_joins.append(q_mscn_bitmap_tpch_join)

for i in range(0, 7):
    with open(f'results/table joins/multi_joins/mscn_bitmap/q_mscn_bitmap_tpch_{i}.csv', 'w') as f:
        for q in mscn_bitmap_tpch_joins[i]:
            f.write(str(q) + '\n')

anova_mscn_tpch = stats.f_oneway(mscn_bitmap_tpch_joins[0], mscn_bitmap_tpch_joins[1],
                                 mscn_bitmap_tpch_joins[2], mscn_bitmap_tpch_joins[3], mscn_bitmap_tpch_joins[4],
                                 mscn_bitmap_tpch_joins[5], mscn_bitmap_tpch_joins[6])

# xgb
xgb_imdb_joins = []
for i in range(0, 3):
    xgb_imdb_join = pd.read_csv(f'results/table joins/multi_joins/xgb/predictions_xgb_imdb_join_{i}.csv')
    q_xgb_imdb_join = print_qerror(xgb_imdb_join.iloc[:, 0], xgb_imdb_join.iloc[:, 1])
    xgb_imdb_joins.append(q_xgb_imdb_join)

for i in range(0, 3):
    with open(f'results/table joins/multi_joins/xgb/q_xgb_imdb_{i}.csv', 'w') as f:
        for q in xgb_imdb_joins[i]:
            f.write(str(q) + '\n')

anova_xgb_imdb = stats.f_oneway(xgb_imdb_joins[0], xgb_imdb_joins[1],
                                xgb_imdb_joins[2])

xgb_tpch_joins = []
for i in range(0, 3):
    xgb_tpch_join = pd.read_csv(f'results/table joins/multi_joins/xgb/predictions_xgb_tpch_join_{i}.csv')
    q_xgb_tpch_join = print_qerror(xgb_tpch_join.iloc[:, 0], xgb_tpch_join.iloc[:, 1])
    xgb_tpch_joins.append(q_xgb_tpch_join)

for i in range(0, 3):
    with open(f'results/table joins/multi_joins/xgb/q_xgb_tpch_{i}.csv', 'w') as f:
        for q in xgb_tpch_joins[i]:
            f.write(str(q) + '\n')

anova_xgb_tpch = stats.f_oneway(xgb_tpch_joins[0], xgb_tpch_joins[1],
                                xgb_tpch_joins[2])

# deepdb
deepdb_imdb_joins = []
for i in range(0, 5):
    deepdb_imdb_join = pd.read_csv(f'results/table joins/multi_joins/deepdb/result_deepdb_imdb_{i}.csv',
                                   usecols=['cardinality_predict', 'cardinality_true'])
    q_deepdb_imdb_join = print_qerror(deepdb_imdb_join.iloc[:, 0], deepdb_imdb_join.iloc[:, 1])
    deepdb_imdb_joins.append(q_deepdb_imdb_join)

for i in range(0, 5):
    with open(f'results/table joins/multi_joins/deepdb/q_deepdb_imdb_{i}.csv', 'w') as f:
        for q in deepdb_imdb_joins[i]:
            f.write(str(q) + '\n')

anova_deepdb_imdb = stats.f_oneway(deepdb_imdb_joins[0], deepdb_imdb_joins[1],
                                   deepdb_imdb_joins[2], deepdb_imdb_joins[3], deepdb_imdb_joins[4])

deepdb_tpch_joins = []
for i in range(0, 7):
    deepdb_tpch_join = pd.read_csv(f'results/table joins/multi_joins/deepdb/result_deepdb_tpch_{i}.csv',
                                   usecols=['cardinality_predict', 'cardinality_true'])
    q_deepdb_tpch_join = print_qerror(deepdb_tpch_join.iloc[:, 0], deepdb_tpch_join.iloc[:, 1])
    deepdb_tpch_joins.append(q_deepdb_tpch_join)

for i in range(0, 7):
    with open(f'results/table joins/multi_joins/deepdb/q_deepdb_tpch_{i}.csv', 'w') as f:
        for q in deepdb_tpch_joins[i]:
            f.write(str(q) + '\n')

anova_deepdb_tpch = stats.f_oneway(deepdb_tpch_joins[0], deepdb_tpch_joins[1],
                                   deepdb_tpch_joins[2], deepdb_tpch_joins[3], deepdb_tpch_joins[4],
                                   deepdb_tpch_joins[5], deepdb_tpch_joins[6])

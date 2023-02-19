import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
from scipy import stats

pio.renderers.default = "browser"


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] == 0:
            preds_unnorm[i] = 1
        if float(labels_unnorm[i]) == 0:
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


# for model performance on whole table
mscn_bfp = pd.read_csv('results/Single table/Overall/predictions_mscn_test_bfp.csv', header=None)
mscn_bfp_bitmap = pd.read_csv('results/Single table/Overall/predictions_mscn_test_bfp_bitmap.csv', header=None)
mscn_fps = pd.read_csv('results/Single table/Overall/predictions_mscn_test_fps.csv', header=None)
mscn_fps_bitmap = pd.read_csv('results/Single table/Overall/predictions_mscn_test_fps_bitmap.csv', header=None)
multi_hist_bfp = pd.read_csv('results/Single table/Overall/bfp_all_multi_hist_5000.csv')
multi_hist_fps = pd.read_csv('results/Single table/Overall/fps_all_multi_hist_5000.csv')
postgre_bfp = pd.read_csv('results/Single table/Overall/predicitons_postgre_bfp.csv')
postgre_fps = pd.read_csv('results/Single table/Overall/predicitons_postgre_fps.csv')
sampling_bfp = pd.read_csv('results/Single table/Overall/predicitons_sampling_bfp.csv')
sampling_fps = pd.read_csv('results/Single table/Overall/predicitons_sampling_fps.csv')
xgb_bfp = pd.read_csv('results/Single table/Overall/predictions_xgb_bfp.csv', header=None)
xgb_fps = pd.read_csv('results/Single table/Overall/predictions_xgb_fps.csv', header=None)
deepdb_bfp = pd.read_csv('results/Single table/Overall/result_deepdb_black_friday_purchase.csv',
                         usecols=['cardinality_predict', 'cardinality_true'])
deepdb_fps = pd.read_csv('results/Single table/Overall/result_deepdb_fps.csv',
                         usecols=['cardinality_predict', 'cardinality_true'])

# performance of each model on different data
#mscn
q_mscn_bfp = print_qerror(mscn_bfp.iloc[:, 0], mscn_bfp.iloc[:, 1])
q_mscn_bfp_bitmap = print_qerror(mscn_bfp_bitmap.iloc[:, 0], mscn_bfp_bitmap.iloc[:, 1])
q_mscn_fps = print_qerror(mscn_fps.iloc[:, 0], mscn_fps.iloc[:, 1])
q_mscn_fps_bitmap = print_qerror(mscn_fps_bitmap.iloc[:, 0], mscn_fps_bitmap.iloc[:, 1])

q_mscn = pd.DataFrame({'q_mscn_bfp': q_mscn_bfp, 'q_mscn_bfp_bitmap': q_mscn_bfp_bitmap, 'q_mscn_fps':q_mscn_fps,
              'q_mscn_fps_bitmap ': q_mscn_fps_bitmap})
q_mscn.to_csv('results/q_mscn.csv', index=False)

mscn_ftest = stats.f_oneway(q_mscn_bfp,q_mscn_fps)
mscn_bitmap_ftest = stats.f_oneway(q_mscn_bfp_bitmap,q_mscn_fps_bitmap)

# postgre
q_postgre_bfp = print_qerror(postgre_bfp.iloc[:, 0], postgre_bfp.iloc[:, 1])
q_postgre_fps = print_qerror(postgre_fps.iloc[:, 0], postgre_fps.iloc[:, 1])
q_postgre = pd.DataFrame({'q_postgre_bfp': q_postgre_bfp, 'q_postgre_fps': q_postgre_fps})
q_postgre.to_csv('results/q_postgre.csv', index=False)
postgre_f = stats.f_oneway(q_postgre_bfp, q_postgre_fps)


# Random Sampling
q_sampling_bfp= print_qerror(sampling_bfp.iloc[:, 0], sampling_bfp.iloc[:, 1])
q_sampling_fps = print_qerror(sampling_fps.iloc[:, 0], sampling_fps.iloc[:, 1])
q_sampling = pd.DataFrame({'q_sampling_bfp': q_sampling_bfp, 'q_sampling_fps': q_sampling_fps})
q_sampling.to_csv('results/q_sampling.csv', index=False)
sampling_f = stats.f_oneway(q_sampling_bfp, q_sampling_fps)

#Multihist
q_multi_hist_bfp= print_qerror(multi_hist_bfp.iloc[:, 0], multi_hist_bfp.iloc[:, 1])
q_multi_hist_fps = print_qerror(multi_hist_fps.iloc[:, 0], multi_hist_fps.iloc[:, 1])
q_multi_hist = pd.DataFrame({'q_multi_hist_bfp': q_multi_hist_bfp, 'q_multi_hist_fps': q_multi_hist_fps})
q_multi_hist.to_csv('results/q_multi_hist.csv', index=False)
multi_hist_f = stats.f_oneway(q_multi_hist_bfp,q_multi_hist_fps)


#Xgboost
q_xgb_bfp= print_qerror(xgb_bfp.iloc[:, 0], xgb_bfp.iloc[:, 1])
q_xgb_fps = print_qerror(xgb_fps.iloc[:, 0], xgb_fps.iloc[:, 1])
q_xgb = pd.DataFrame({'q_xgb_bfp': q_xgb_bfp, 'q_xgb_fps': q_xgb_fps})
q_xgb.to_csv('results/q_xgb.csv', index=False)
xgb_f = stats.f_oneway(q_xgb_bfp,q_xgb_fps)


#deepdb
q_deepdb_bfp= print_qerror(deepdb_bfp.iloc[:, 0], deepdb_bfp.iloc[:, 1])
q_deepdb_fps = print_qerror(deepdb_fps.iloc[:, 0], deepdb_fps.iloc[:, 1])
q_deepdb = pd.DataFrame({'q_deepdb_bfp': q_deepdb_bfp, 'q_deepdb_fps': q_deepdb_fps})
q_deepdb.to_csv('results/q_deepdb.csv', index=False)
deepdb_f = stats.f_oneway(q_deepdb_bfp, q_deepdb_fps)

# for 2-col correlations
# bfp
i = 1
postgre_bfp_2col=[]
sampling_bfp_2col=[]
multi_hist_bfp_2col=[]
mscn_bfp_2col=[]
xgb_bfp_2col=[]
deepdb_bfp_2col=[]
while i<8:
    mscn_bfp = pd.read_csv(f"results/Single table/2col/mscn/predictions_mscn_test_bfp_2col_{i}.csv", header=None)
    postgre_bfp = pd.read_csv(f"results/Single table/2col/postgre/predicitons_postgre_bfp_2col_{i}.csv")
    multi_hist_bfp = pd.read_csv(f"results/Single table/2col/multi_hist/multi_hist_bfp_2col_{i}.csv")
    xgb_bfp = pd.read_csv(f"results/Single table/2col/xgb/predictions_xgb_bfp_2col_{i}.csv", header=None)
    deepdb_bfp = pd.read_csv(f"results/Single table/2col/deepdb/result_deepdb_bfp_2col_{i}.csv",
                             usecols=['cardinality_predict', 'cardinality_true'])
    sampling_bfp = pd.read_csv(f"results/Single table/2col/random_sampling/predicitons_sampling_bfp_2col_{i}.csv")
    mscn_bfp_2col.append(mscn_bfp)
    postgre_bfp_2col.append(postgre_bfp)
    sampling_bfp_2col.append(sampling_bfp)
    multi_hist_bfp_2col.append(multi_hist_bfp)
    xgb_bfp_2col.append(xgb_bfp)
    deepdb_bfp_2col.append(deepdb_bfp)
    i = i + 2

j = 1
q_postgre_2col =pd.DataFrame({})
for prediction in postgre_bfp_2col:
    q_postgre_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_postgre_2col.to_csv('results/q_postgre_2col.csv', index=False)

j = 1
q_sampling_2col = pd.DataFrame({})
for prediction in sampling_bfp_2col:
    q_sampling_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_sampling_2col.to_csv('results/q_sampling_2col.csv', index=False)

j = 1
q_multi_hist_2col = pd.DataFrame({})
for prediction in multi_hist_bfp_2col:
    q_multi_hist_2col[str(j)] = print_qerror(prediction.iloc[:, 1], prediction.iloc[:, 2])
    j = j+1
q_multi_hist_2col.to_csv('results/q_multi_hist_2col.csv', index=False)

j = 1
q_mscn_2col = pd.DataFrame({})
for prediction in mscn_bfp_2col:
    q_mscn_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_mscn_2col.to_csv('results/q_mscn_2col.csv', index=False)

j = 1
q_xgb_2col = pd.DataFrame({})
for prediction in xgb_bfp_2col:
    q_xgb_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_xgb_2col.to_csv('results/q_xgb_2col.csv', index=False)

j = 1
q_deepdb_2col = pd.DataFrame({})
for prediction in deepdb_bfp_2col:
    q_deepdb_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_deepdb_2col.to_csv('results/q_deepdb_2col.csv', index=False)

# fps
i = 1
postgre_fps_2col=[]
sampling_fps_2col=[]
multi_hist_fps_2col=[]
mscn_fps_2col=[]
xgb_fps_2col=[]
deepdb_fps_2col=[]
while i<=10:
    mscn_fps = pd.read_csv(f"results/Single table/2col/mscn/predictions_mscn_test_fps_2col_{i}.csv", header=None)
    postgre_fps = pd.read_csv(f"results/Single table/2col/postgre/predicitons_postgre_fps_2col_{i}.csv")
    multi_hist_fps = pd.read_csv(f"results/Single table/2col/multi_hist/multi_hist_fps_2col_{i}.csv")
    xgb_fps = pd.read_csv(f"results/Single table/2col/xgb/predictions_xgb_fps_2col_{i}.csv", header=None)
    deepdb_fps = pd.read_csv(f"results/Single table/2col/deepdb/result_deepdb_fps_2col_{i}.csv",
                             usecols=['cardinality_predict', 'cardinality_true'])
    sampling_fps = pd.read_csv(f"results/Single table/2col/random_sampling/predicitons_sampling_fps_2col_{i}.csv")
    mscn_fps_2col.append(mscn_fps)
    postgre_fps_2col.append(postgre_fps)
    sampling_fps_2col.append(sampling_fps)
    multi_hist_fps_2col.append(multi_hist_fps)
    xgb_fps_2col.append(xgb_fps)
    deepdb_fps_2col.append(deepdb_fps)
    i = i + 1

j = 1
q_postgre_2col =pd.DataFrame({})
for prediction in postgre_fps_2col:
    q_postgre_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_postgre_2col.to_csv('results/q_postgre_2col.csv', index=False)

j = 1
q_sampling_2col = pd.DataFrame({})
for prediction in sampling_fps_2col:
    q_sampling_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_sampling_2col.to_csv('results/q_sampling_2col.csv', index=False)

j = 1
q_multi_hist_2col = pd.DataFrame({})
for prediction in multi_hist_fps_2col:
    q_multi_hist_2col[str(j)] = print_qerror(prediction.iloc[:, 1], prediction.iloc[:, 2])
    j = j+1
q_multi_hist_2col.to_csv('results/q_multi_hist_2col.csv', index=False)

j = 1
q_mscn_2col = pd.DataFrame({})
for prediction in mscn_fps_2col:
    q_mscn_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_mscn_2col.to_csv('results/q_mscn_2col.csv', index=False)

j = 1
q_xgb_2col = pd.DataFrame({})
for prediction in xgb_fps_2col:
    q_xgb_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_xgb_2col.to_csv('results/q_xgb_2col.csv', index=False)

j = 1
q_deepdb_2col = pd.DataFrame({})
for prediction in deepdb_fps_2col:
    q_deepdb_2col[str(j)] = print_qerror(prediction.iloc[:, 0], prediction.iloc[:, 1])
    j = j+1
q_deepdb_2col.to_csv('results/q_deepdb_2col.csv', index=False)


















bfp_all_qerror = print_qerror(mscn_bfp.iloc[:, 0], mscn_bfp.iloc[:, 1])
fig = go.Figure()
fig.add_trace(go.Box(x=bfp_all_qerror))
fig.update_traces(q1=[np.percentile(bfp_all_qerror, 25)], median=[np.median(bfp_all_qerror)],
                  q3=[np.percentile(bfp_all_qerror, 99)],
                  lowerfence=[np.min(bfp_all_qerror)],
                  upperfence=[np.max(bfp_all_qerror)], mean=[np.mean(bfp_all_qerror)])  # , width=0.1, height=600)
fig.update_layout(height=600, width=300)
fig.show()

with open('results/mscn_bfp_all_qerror.csv', 'w') as f:
    for q in bfp_all_qerror:
        f.write(str(q))
        f.write('\n')
    f.close()

import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np

pio.renderers.default = "browser"
result_mscn_bfp = pd.read_csv('results/predictions_mscn_test_bfp.csv', header=None)


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


bfp_all_qerror = print_qerror(result_mscn_bfp.iloc[:, 0], result_mscn_bfp.iloc[:, 1])
fig = go.Figure()
fig.add_trace(go.Box(x=bfp_all_qerror))
fig.update_traces(q1=[np.percentile(bfp_all_qerror, 25)], median=[np.median(bfp_all_qerror)],
                  q3=[np.percentile(bfp_all_qerror, 99)],
                  lowerfence=[np.min(bfp_all_qerror)],
                  upperfence=[np.max(bfp_all_qerror)], mean=[np.mean(bfp_all_qerror)]) # , width=0.1, height=600)
fig.update_layout(height=600, width=300)
fig.show()

with open('results/mscn_bfp_all_qerror.csv','w') as f:
    for q in bfp_all_qerror:
        f.write(str(q))
        f.write('\n')
    f.close()


import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
from scipy import stats
import math
import time
from tqdm import tqdm

class TreeEnsemble():

    def __init__(self):
        self.model = xgb.Booster({'nthread': 10})

    def train(self, train_data, labels, num_round=10,
              param={'max_depth': 5, 'eta': 0.1, 'booster': 'gbtree', 'objective': 'reg:logistic'}):
        print(train_data.shape, labels.shape)
        train_len = int(0.8 * len(train_data))
        dtrain = xgb.DMatrix(train_data[:train_len], label=labels[:train_len])
        dvalidate = xgb.DMatrix(train_data[train_len:], label=labels[train_len:])
        evallist = [(dvalidate, 'test'), (dtrain, 'train')]
        self.model = xgb.train(param, dtrain, num_round, evallist)

    def save_model(self, path):
        self.model.save_model(path + '.xgb.model')

    def load_model(self, path):
        self.model.load_model(path + '.xgb.model')

    def estimate(self, test_data):
        dtest = xgb.DMatrix(test_data)
        return self.model.predict(dtest)


def normalize(x, min_card_log, max_card_log):
    return np.maximum(np.minimum((np.log(x) - min_card_log) / (max_card_log - min_card_log), 1.0), 0.0)


def unnormalize(x, min_card_log, max_card_log):
    return np.exp(x * (max_card_log - min_card_log) + min_card_log)


def prepare_pattern_workload(path, min_max_file):
    pattern2training = {}
    pattern2truecard = {}
    minmax = pd.read_csv(min_max_file)
    minmax = minmax.set_index('name')
    min_card_log = 999999999999.0
    max_card_log = 0.0
    with open(path + '.csv', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            tables = sorted([x.split(' ')[1] for x in line.split('#')[0].split(',')])
            local_cols = []
            vecs = []
            for col_name in minmax.index:
                if col_name.split('.')[0] in tables:
                    local_cols.append(col_name)
                    vecs.append(0.0)
                    vecs.append(1.0)
            conds = [x for x in line.split('#')[2].split(',')]
            for i in range(int(len(conds) / 3)):
                attr = conds[i * 3]
                op = conds[i * 3 + 1]
                value = conds[i * 3 + 2]
                idx = local_cols.index(attr)
                maximum = float(minmax.loc[attr]['max'])
                minimum = float(minmax.loc[attr]['min'])
                distinct_num = minmax.loc[attr]['num_unique_values']
                if op == '=':
                    offset = (maximum - minimum) / distinct_num / 2.0
                    upper = ((float(value) + offset) - minimum) / (maximum - minimum)
                    lower = (float(value) - offset - minimum) / (maximum - minimum)
                elif op == '<':
                    upper = (float(value) - minimum) / (maximum - minimum)
                    lower = 0.0
                elif op == '>':
                    upper = 1.0
                    lower = (float(value) - minimum) / (maximum - minimum)
                else:
                    raise Exception(op)
                if upper < vecs[idx * 2 + 1]:
                    vecs[idx * 2 + 1] = upper
                if lower > vecs[idx * 2]:
                    vecs[idx * 2] = lower
            key = '_'.join(tables)
            card = float(line.split('#')[-1])
            if card == 0:
                card = 1
            if key in pattern2training:
                pattern2training[key].append(vecs)
                pattern2truecard[key].append(card)
            else:
                pattern2training[key] = [vecs]
                pattern2truecard[key] = [card]
            if math.log(card) < min_card_log:
                min_card_log = math.log(card)
            if math.log(card) > max_card_log:
                max_card_log = math.log(card)

    return pattern2training, pattern2truecard, min_card_log, max_card_log


def train_for_all_pattern(path, min_max_file):
    pattern2training, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path, min_max_file)
    print('min_card_log: {}, max_card_log: {}'.format(min_card_log, max_card_log))
    pattern2model = {}
    for k, v in pattern2training.items():
        print(k, len(v), len(v[0]))
        print(v[0])
        print(v[1])
        pattern2model[k] = TreeEnsemble()
        pattern2model[k].train(np.array(v), normalize(pattern2truecard[k], min_card_log, max_card_log), num_round=100)
    with open('XGboost/xgb.model', 'wb') as f:
        pickle.dump(pattern2model, f)
    return pattern2model


def print_qerror(preds_unnorm, labels_unnorm):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    '''fmetric.write("Median: {}".format(np.median(qerror))+ '\n'+ "90th percentile: {}".format(np.percentile(qerror, 90))+ '\n'+ "95th percentile: {}".format(np.percentile(qerror, 95))+\
            '\n'+ "99th percentile: {}".format(np.percentile(qerror, 99))+ '\n'+ "99th percentile: {}".format(np.percentile(qerror, 99))+ '\n'+ "Max: {}".format(np.max(qerror))+ '\n'+\
            "Mean: {}".format(np.mean(qerror))+ '\n')'''
    print("Median: {}".format(np.median(qerror)))
    print("90th percentile: {}".format(np.percentile(qerror, 90)))
    print("95th percentile: {}".format(np.percentile(qerror, 95)))
    print("99th percentile: {}".format(np.percentile(qerror, 99)))
    print("Max: {}".format(np.max(qerror)))
    print("Mean: {}".format(np.mean(qerror)))


def print_mse(preds_unnorm, labels_unnorm):
    # fmetric.write("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean())+ '\n')
    print("MSE: {}".format(((preds_unnorm - labels_unnorm) ** 2).mean()))


def print_mape(preds_unnorm, labels_unnorm):
    # fmetric.write("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100)+ '\n')
    print("MAPE: {}".format(((np.abs(preds_unnorm - labels_unnorm) / labels_unnorm)).mean() * 100))


def print_pearson_correlation(x, y):
    PCCs = stats.pearsonr(x, y)
    # fmetric.write("Pearson Correlation: {}".format(PCCs)+ '\n')
    print("Pearson Correlation: {}".format(PCCs))


def test_for_all_pattern(path, workload, pattern2model, min_max_file):
    pattern2testing, pattern2truecard, min_card_log, max_card_log = prepare_pattern_workload(path, min_max_file)
    print('min_card_log: {}, max_card_log: {}'.format(min_card_log, max_card_log))
    cards = []
    true_cards = []
    start = time.time()
    for k, v in pattern2testing.items():
        model = pattern2model[k]
        cards += unnormalize(model.estimate(np.array(v)), min_card_log, max_card_log).tolist()
        true_cards += pattern2truecard[k]
    end = time.time()
    # fmetric.write("Prediction Time {}ms for each of {} queries".format((end - start) / len(cards) * 1000, len(cards))+'\n')
    print("Prediction Time {}ms for each of {} queries".format((end - start) / len(cards) * 1000, len(cards)))
    print_qerror(np.array(cards), np.array(true_cards))
    print_mse(np.array(cards), np.array(true_cards))
    print_mape(np.array(cards), np.array(true_cards))
    print_pearson_correlation(np.array(cards), np.array(true_cards))
    with open(f'results/predictions_xgb_{workload}.csv', 'w') as f:
        for i in range(len(cards)):
            f.write(f'{cards[i]},{true_cards[i]}')
            f.write('\n')


import pandas as pd
import numpy as np
from tqdm import tqdm

class Partition(object):

    def __init__(self, df):
        self.df = df
        # a list of tuples (low, high)
        self.columns = df.columns
        self.boundaries = {} # col:[low,high]
        #self.density = None
        self.n_total = len(df)
        self.set_boundaries()
        self.col_distinct = {}


    def set_boundaries(self):
        for col in self.df.columns:
            self.boundaries[col] = [self.df[col].min(), self.df[col].max()]
        return self.boundaries

    def margin(self, col):
        if self.df is None:
            print('please recreate partition')
        else:
            if col in self.columns:
                margin = self.df[col].value_counts().sort_index()
                return margin
            else:
                print(f'{col} is not a correct column')

    #def density(self):
    #    dist = 1
    #    for col in self.df.columns:
    #        dist = dist * len(self.df[col].value_counts())
    #    self.density = self.n_total/dist
    #    return self.density

    def condense(self):
        for col in self.columns:
            self.col_distinct[col] = self.df[col].unique()
        self.df = None


def compute_maxdiff(partition, maxdiff_map):
    for col in partition.columns:
        counter = partition.margin(col)
        split_vals = None
        # compute Diff(V, A)
        if len(counter) == 1:  # 若该列只有一个值，则该dimension无法再分
            maxdiff = 0
        else:
            spread = counter.index[1:] - counter.index[:-1]
            spread_m_counts = spread * counter.iloc[:-1]
            spread_m_counts = pd.concat([spread_m_counts.to_frame(),
                        pd.DataFrame({counter.iloc[-1] * spread[-1]}, index=[counter.index[-1]])])
            spread_m_counts = abs(spread_m_counts.values[1:] - spread_m_counts.values[:-1])
            maxdiff = 0
            if len(spread_m_counts) > 0:
                maxdiff = max(spread_m_counts.max(), 0)
                if maxdiff>0:
                    position = list(spread_m_counts).index(maxdiff)
                    split_vals = (counter.index[position],counter.index[position+1])
        if maxdiff not in maxdiff_map:
            maxdiff_map[maxdiff] = [[partition, col, split_vals]]
        else:
            maxdiff_map[maxdiff].append([partition, col, split_vals])
        #partition_to_maxdiff[partition].add([col, maxdiff, split_vals])


def next_partition(maxdiff_map):
    maxdiff, candidates = sorted(maxdiff_map.items(), key=lambda x: x[0])[-1]
    candidate = candidates[0]
    if len(candidates)==1:
        maxdiff_map.pop(maxdiff)

    delete_list_m = []
    for m in maxdiff_map:
        delete_list_p = []
        for p_list in maxdiff_map[m]:
            if p_list[0] == candidate[0]:
                delete_list_p.append(p_list)
        for d_p in delete_list_p:
            maxdiff_map[m].remove(d_p)
        if len(maxdiff_map[m])==0:
            delete_list_m.append(m)

    for d_m in delete_list_m:
        maxdiff_map.pop(d_m)

    return candidate, maxdiff_map, maxdiff


def create_new_partition(partition, col, split_vals):
    new_p1 = Partition(partition.df[partition.df[col]<=split_vals[0]])
    new_p2 = Partition(partition.df[partition.df[col]>=split_vals[1]])
    return [new_p1,new_p2]


def bulid_hist(df, limit):
    maxdiff_map = {}  # maxdiff: [[partition, col, split_vals],...]
    #partition_to_maxdiff = {}  # partition: [[col, maxdiff, split_vals],...]
    p = Partition(df)
    partitions = [p]
    i = 1
    if limit>1:
        compute_maxdiff(p, maxdiff_map)
        candidate, maxdiff_map, maxdiff = next_partition(maxdiff_map)
        if maxdiff == 0:
            print('no partition space remained')
            print('Condense partitions')
            for p in tqdm(partitions):
                p.condense()
            return partitions
        else:
            new_partitions = create_new_partition(candidate[0], candidate[1], candidate[2])
            partitions.append(new_partitions[0])
            partitions.append(new_partitions[1])
            partitions.remove(candidate[0])
            del candidate[0]
        i = i+1
        print(f'partition number {len(partitions)}')
    else:
        print('partition number = 1')
        print('Condense partitions')
        for p in tqdm(partitions):
            p.condense()
        return partitions

    while i < limit:
        for n_p in new_partitions:
            compute_maxdiff(n_p, maxdiff_map)
        candidate, maxdiff_map, maxdiff = next_partition(maxdiff_map)
        if maxdiff == 0:
            print('no partition space remained')
            print('Condense partitions')
            for p in tqdm(partitions):
                p.condense()
            return partitions
        else:
            new_partitions = create_new_partition(candidate[0], candidate[1], candidate[2])
            partitions.append(new_partitions[0])
            partitions.append(new_partitions[1])
            partitions.remove(candidate[0])
            del candidate[0]
        i=i+1
        if (i <= 10) or (i % 1000 == 0):
            print(f'partition number {len(partitions)}')

    print('Condense partitions')
    for p in tqdm(partitions):
        p.condense()

    return partitions



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



def query_evaluate(sql_csv_file, partitions):
    with open(sql_csv_file) as f:
        lines = f.readlines()
        pred_card = []
        true_card = []
        for line in tqdm(lines):
            predicates = line.split('#')[2].split(',')
            card = 0
            for p in partitions:
                i = 0
                frac = 1
                while i < len(predicates):
                    col_distinct = p.col_distinct[predicates[i].split('.')[1]]
                    if predicates[i + 1] == '=':
                        predicates[i + 1] = '=='
                    covered = sum(eval(f'v{predicates[i+1]}{predicates[i+2]}') for v in col_distinct)
                    fraction = covered/len(col_distinct)
                    if covered == 0:
                        frac = 0
                        break
                    else:
                        frac = frac * fraction
                    i = i+3
                card = card + (p.n_total * frac)
            if 1 > card > 0:
                card = 1
            pred_card.append(card)
            true_card.append(int(line.split('#')[-1]))
        print_qerror(pred_card, true_card)
    return pred_card, true_card


import bisect
import collections
import operator
import time
import numpy as np
import MultiHist.common as common
import pandas as pd


def LoadMyDataset(filepath, name_table, usecols):
    # Make sure that this loads data correctly.
    df = pd.read_csv(filepath, usecols=usecols)
    df.columns = [c.lower() for c in df.columns]
    return common.CsvTable(name_table, df, cols=df.columns)


class CardEst(object):
    """Base class for a cardinality estimator."""

    def __init__(self):
        self.query_starts = []
        self.query_dur_ms = []
        self.errs = []
        self.est_cards = []
        self.true_cards = []

        self.name = 'CardEst'

    def Query(self, columns, operators, vals):
        """Estimates cardinality with the specified conditions.

        Args:
            columns: list of Column objects to filter on.
            operators: list of string representing what operation to perform on
              respective columns; e.g., ['<', '>='].
            vals: list of raw values to filter columns on; e.g., [50, 100000].
              These are not bin IDs.
        Returns:
            Predicted cardinality.
        """
        raise NotImplementedError

    def OnStart(self):
        self.query_starts.append(time.time())

    def OnEnd(self):
        self.query_dur_ms.append((time.time() - self.query_starts[-1]) * 1e3)

    def AddError(self, err):
        self.errs.append(err)

    def AddError(self, err, est_card, true_card):
        self.errs.append(err)
        self.est_cards.append(est_card)
        self.true_cards.append(true_card)

    def __str__(self):
        return self.name

    def get_stats(self):
        return [
            self.query_starts, self.query_dur_ms, self.errs, self.est_cards,
            self.true_cards
        ]

    def merge_stats(self, state):
        self.query_starts.extend(state[0])
        self.query_dur_ms.extend(state[1])
        self.errs.extend(state[2])
        self.est_cards.extend(state[3])
        self.true_cards.extend(state[4])

    def report(self):
        est = self
        print(est.name, "max", np.max(est.errs), "99th",
              np.quantile(est.errs, 0.99), "95th", np.quantile(est.errs, 0.95),
              "median", np.quantile(est.errs, 0.5), "time_ms",
              np.mean(est.query_dur_ms))


class MaxDiffHistogram(CardEst):
    """MaxDiff n-dimensional histogram."""

    def __init__(self, table, limit):
        super(MaxDiffHistogram, self).__init__()
        self.table = table
        self.limit = limit
        self.partitions = []
        self.maxdiff = {}
        self.partition_to_maxdiff = {}
        self.num_new_partitions = 2
        # map<cid, map<bound_type, map<bound_value, list(partition id)>>>
        self.column_bound_map = {}
        for cid in range(len(self.table.columns)):
            self.column_bound_map[cid] = {}
            self.column_bound_map[cid]['l'] = {}
            self.column_bound_map[cid]['u'] = {}
        # map<cid, map<bound_type, sorted_list(bound_value)>>
        self.column_bound_index = {}
        for cid in range(len(self.table.columns)):
            self.column_bound_index[cid] = {}
            self.column_bound_index[cid]['l'] = []
            self.column_bound_index[cid]['u'] = []
        self.name = str(self)

        print('Building MaxDiff histogram, may take a while...')
        self._build_histogram()

    def __str__(self):
        return 'maxdiff[{}]'.format(self.limit)

    class Partition(object):

        def __init__(self):
            # a list of tuples (low, high)
            self.boundaries = []
            # a list of row id that belongs to this partition
            self.data_points = []
            # per-column uniform spread
            self.uniform_spreads = []
            # per-distinct value density
            self.density = None
            self.col_value_list = {}
            self.rowid_to_position = {}

        def Size(self):
            total_size = 0
            for _ in self.uniform_spreads:
                total_size += (4 * len(_))
            total_size += 4
            return total_size

    def _compute_maxdiff(self, partition):
        for col in range(len(partition.boundaries)):
            vals = partition.col_value_list[col]
            counter_start = time.time()
            counter = pd.Series(vals).value_counts().sort_index()

            # compute Diff(V, A)
            if len(counter) == 1:  # 若该列只有一个值，则该dimension无法再分
                maxdiff = 0
            else:
                spread = counter.index[1:] - counter.index[:-1]
                spread_m_counts = spread * counter.iloc[:-1]
                # my modification
                # spread_m_counts = spread_m_counts.append(pd.Series({counter.index[-1]: counter.iloc[-1] * spread[-1]}))
                pd.concat([spread_m_counts.to_frame(),
                           pd.DataFrame({counter.iloc[-1] * spread[-1]}, index=[counter.index[-1]])])
                spread_m_counts = abs(spread_m_counts.values[1:] - spread_m_counts.values[:-1])
                maxdiff = 0
                if len(spread_m_counts) > 0:
                    maxdiff = max(spread_m_counts.max(), 0)
                    # maxdiff = max(spread_m_counts.values.max(), 0)
            if maxdiff not in self.maxdiff:
                self.maxdiff[maxdiff] = [(partition, col)]
            else:
                self.maxdiff[maxdiff].append((partition, col))
            self.partition_to_maxdiff[partition].add(maxdiff)

    def _build_histogram(self):
        # initial partition
        p = self.Partition()
        # populate initial boundary
        for cid in range(len(self.table.columns)):
            if not self.table.columns[cid].data.dtype == 'int64':
                p.boundaries.append(
                    (0, self.table.columns[cid].distribution_size, True))
            else:
                p.boundaries.append((min(self.table.columns[cid].data),
                                     max(self.table.columns[cid].data), True))
        # include all rowids
        self.table_ds = common.TableDataset(self.table)
        num_rows = self.table.cardinality
        p.data_points = np.arange(num_rows)
        for cid in range(len(p.boundaries)):
            if not self.table.columns[cid].data.dtype == 'int64':
                p.col_value_list[cid] = self.table_ds.tuples_np[:, cid]
            else:
                p.col_value_list[cid] = self.table.columns[cid].data
        p.rowid_to_position = list(np.arange(num_rows))
        self.partition_to_maxdiff[p] = set()
        self._compute_maxdiff(p)
        self.partitions.append(p)

        while len(self.partitions) < self.limit:
            start_next_partition = time.time()
            (split_partition_index, split_column_index, partition_boundaries,
             global_maxdiff) = self.next_partition_candidate(
                self.partitions, len(self.table.columns), self.table,
                min(self.num_new_partitions,
                    self.limit - len(self.partitions) + 1), self.maxdiff)
            print('determining partition number ', len(self.partitions))
            if global_maxdiff == 0:
                print('maxdiff already 0 before reaching bucket limit')
                break
            start_generate_next_partition = time.time()
            new_partitions = self.generate_new_partitions(
                self.partitions[split_partition_index], split_column_index,
                partition_boundaries)
            for p in new_partitions:
                self.partition_to_maxdiff[p] = set()
                self._compute_maxdiff(p)
            for d in self.partition_to_maxdiff[
                self.partitions[split_partition_index]]:
                remove_set = set()
                for cid in range(len(self.table.columns)):
                    remove_set.add(
                        (self.partitions[split_partition_index], cid))
                for tp in remove_set:
                    if tp in self.maxdiff[d]:
                        self.maxdiff[d].remove(tp)
                if len(self.maxdiff[d]) == 0:
                    del self.maxdiff[d]
            del self.partition_to_maxdiff[
                self.partitions[split_partition_index]]
            self.partitions.pop(split_partition_index)
            self.partitions += new_partitions
        # finish partitioning, for each partition we condense its info and
        # compute uniform spread.
        total_point = 0
        for pid, partition in enumerate(self.partitions):
            total_point += len(partition.data_points)
            total = len(partition.data_points)
            total_distinct = 1
            for cid, boundary in enumerate(partition.boundaries):
                distinct = len(
                    set(self.table.columns[cid].data[rowid]
                        for rowid in partition.data_points))
                if distinct == 1:
                    if not self.table.columns[cid].data.dtype == 'int64':
                        partition.uniform_spreads.append([
                            list(
                                set(self.table_ds.
                                    tuples_np[partition.data_points, cid]))[0]
                        ])
                    else:
                        partition.uniform_spreads.append([
                            list(
                                set(self.table.columns[cid].data[
                                        partition.data_points]))[0]
                        ])
                else:
                    uniform_spread = None
                    spread_length = None
                    if boundary[2]:
                        spread_length = float(
                            (boundary[1] - boundary[0])) / (distinct - 1)
                        uniform_spread = [boundary[0]]
                    else:
                        spread_length = float(
                            (boundary[1] - boundary[0])) / (distinct)
                        uniform_spread = [boundary[0] + spread_length]
                    for _ in range(distinct - 2):
                        uniform_spread.append(uniform_spread[-1] +
                                              spread_length)
                    uniform_spread.append(boundary[1])
                    partition.uniform_spreads.append(uniform_spread)
                total_distinct = total_distinct * distinct
            partition.density = float(total) / total_distinct
        print('total number of point is ', total_point)

        # populate column bound map and list metadata
        for cid in range(len(self.table.columns)):
            for pid, partition in enumerate(self.partitions):
                if partition.boundaries[cid][0] not in self.column_bound_map[
                    cid]['l']:
                    self.column_bound_map[cid]['l'][partition.boundaries[cid]
                    [0]] = [pid]
                else:
                    self.column_bound_map[cid]['l'][partition.boundaries[cid]
                    [0]].append(pid)

                if partition.boundaries[cid][1] not in self.column_bound_map[
                    cid]['u']:
                    self.column_bound_map[cid]['u'][partition.boundaries[cid]
                    [1]] = [pid]
                else:
                    self.column_bound_map[cid]['u'][partition.boundaries[cid]
                    [1]].append(pid)

                self.column_bound_index[cid]['l'].append(
                    partition.boundaries[cid][0])
                self.column_bound_index[cid]['u'].append(
                    partition.boundaries[cid][1])
            self.column_bound_index[cid]['l'] = sorted(
                set(self.column_bound_index[cid]['l']))
            self.column_bound_index[cid]['u'] = sorted(
                set(self.column_bound_index[cid]['u']))

    def next_partition_candidate(self, partitions, column_number, table,
                                 num_new_partitions, maxdiff_map):
        global_maxdiff = max(sorted(maxdiff_map.keys()))
        partition, cid = maxdiff_map[global_maxdiff][0]
        vals = partition.col_value_list[cid]
        counter = collections.Counter(vals)
        first_key = True
        prev_key = None
        diff = []
        spread = 1                                           # my modification 我加的
        for key in sorted(counter.keys()):
            if first_key:
                first_key = False
                prev_key = key
            else:
                spread = key - prev_key
                diff.append((prev_key, (spread * counter[prev_key])))
                prev_key = key
        diff.append((prev_key, (spread * counter[prev_key]))) # my modification calculate area for all points
        d = list()
        for i in range(len(diff) - 1):
            d.append((diff[i][0], abs(diff[i + 1][1] - diff[i][1])))
        diff = d
        partition_boundaries = sorted(
            list(
                tp[0]
                for tp in sorted(diff, key=operator.itemgetter(
                    1), reverse=True)[:min(num_new_partitions - 1, len(diff))]))
        return (partitions.index(partition), cid, partition_boundaries,
                global_maxdiff)

    def generate_new_partitions(self, partition, partition_column_index,
                                partition_boundaries):
        new_partitions = []
        for i in range(len(partition_boundaries) + 1):
            new_partition = self.Partition()
            for cid, boundary in enumerate(partition.boundaries):
                if not cid == partition_column_index:
                    new_partition.boundaries.append(boundary)
                else:
                    if i == 0:
                        new_partition.boundaries.append(
                            (boundary[0], partition_boundaries[i], boundary[2]))
                    elif i == len(partition_boundaries):
                        new_partition.boundaries.append(
                            (partition_boundaries[i - 1], boundary[1], False))
                    else:
                        new_partition.boundaries.append(
                            (partition_boundaries[i - 1],
                             partition_boundaries[i], False))
            new_partitions.append(new_partition)
        # distribute data points to new partitions
        for rowid in partition.data_points:
            if not self.table.columns[
                       partition_column_index].data.dtype == 'int64':
                val = self.table_ds.tuples_np[rowid, partition_column_index]
            else:
                val = self.table.columns[partition_column_index].data[rowid]
            # find the new partition that the row belongs to
            new_partitions[bisect.bisect_left(partition_boundaries,
                                              val)].data_points.append(rowid)

        # debug
        start = time.time()
        for new_partition in new_partitions:
            for cid in range(len(new_partition.boundaries)):
                new_partition.col_value_list[cid] = [
                    partition.col_value_list[cid][
                        partition.rowid_to_position[rowid]]
                    for rowid in new_partition.data_points
                ]
            pos = 0
            for rowid in new_partition.data_points:
                new_partition.rowid_to_position[rowid] = pos
                pos += 1
            if len(new_partition.data_points) == 0:
                print('found partition with no data!')
                print(sorted(
                    list(self.table.columns[partition_column_index].data[rowid]
                         for rowid in partition.data_points)))
                print(partition_boundaries)
        return new_partitions

    def _populate_column_set_map(self, c, o, v, column_set_map):
        cid = self.table.ColumnIndex(c.name)
        column_set_map[cid] = set()
        if o in ['<', '<=']:
            insert_index = None
            if o == '<':
                insert_index = bisect.bisect_left(
                    self.column_bound_index[cid]['l'], v)
                for i in range(insert_index):
                    column_set_map[cid] = column_set_map[cid].union(
                        self.column_bound_map[cid]['l'][
                            self.column_bound_index[cid]['l'][i]])
            else:
                insert_index = bisect.bisect(self.column_bound_index[cid]['l'],
                                             v)
                for i in range(insert_index):
                    if self.column_bound_index[cid]['l'][i] == v:
                        for pid in self.column_bound_map[cid]['l'][v]:
                            if self.partitions[pid].boundaries[cid][2]:
                                # add only when the lower bound is inclusive
                                column_set_map[cid].add(pid)
                    else:
                        column_set_map[cid] = column_set_map[cid].union(
                            self.column_bound_map[cid]['l'][
                                self.column_bound_index[cid]['l'][i]])

        elif o in ['>', '>=']:
            insert_index = None
            if o == '>':
                insert_index = bisect.bisect(self.column_bound_index[cid]['u'],
                                             v)
            else:
                insert_index = bisect.bisect_left(
                    self.column_bound_index[cid]['u'], v)
            for i in range(insert_index,
                           len(self.column_bound_index[cid]['u'])):
                column_set_map[cid] = column_set_map[cid].union(
                    self.column_bound_map[cid]['u'][self.column_bound_index[cid]
                    ['u'][i]])
        else:
            assert o == '=', o
            lower_bound_set = set()
            insert_index = bisect.bisect(self.column_bound_index[cid]['l'], v)
            for i in range(insert_index):
                if self.column_bound_index[cid]['l'][i] == v:
                    for pid in self.column_bound_map[cid]['l'][v]:
                        if self.partitions[pid].boundaries[cid][2]:
                            # add only when the lower bound is inclusive
                            lower_bound_set.add(pid)
                else:
                    lower_bound_set = lower_bound_set.union(
                        self.column_bound_map[cid]['l'][
                            self.column_bound_index[cid]['l'][i]])

            upper_bound_set = set()
            insert_index = bisect.bisect_left(self.column_bound_index[cid]['u'],
                                              v)
            for i in range(insert_index,
                           len(self.column_bound_index[cid]['u'])):
                upper_bound_set = upper_bound_set.union(
                    self.column_bound_map[cid]['u'][self.column_bound_index[cid]
                    ['u'][i]])
            column_set_map[cid] = lower_bound_set.intersection(upper_bound_set)

    def _estimate_cardinality_per_partition(self, partition, columns, operators,
                                            vals):
        distinct_val_covered = 1
        observed_cid = []
        for c, o, v in zip(columns, operators, vals):
            if not c.data.dtype == 'int64':
                v = c.ValToBin(v)
            cid = self.table.ColumnIndex(c.name)
            observed_cid.append(cid)
            spread = partition.uniform_spreads[cid]
            if o in ['<', '<=']:
                if o == '<':
                    distinct_val_covered = distinct_val_covered * bisect.bisect_left(
                        spread, v)
                else:
                    distinct_val_covered = distinct_val_covered * bisect.bisect(
                        spread, v)
            elif o in ['>', '>=']:
                if o == '>':
                    distinct_val_covered = distinct_val_covered * (
                            len(spread) - bisect.bisect(spread, v))
                else:
                    distinct_val_covered = distinct_val_covered * (
                            len(spread) - bisect.bisect_left(spread, v))
            else:
                assert o == '=', o
                if not v in spread:
                    distinct_val_covered = 0
        for cid in range(len(partition.uniform_spreads)):
            if not cid in observed_cid:
                distinct_val_covered = distinct_val_covered * len(
                    partition.uniform_spreads[cid])
        return distinct_val_covered * partition.density

    def Query(self, columns, operators, vals):
        self.OnStart()
        # map<cid, set(pids)>
        column_set_map = {}
        for c, o, v in zip(columns, operators, vals):
            if not c.data.dtype == 'int64':
                v = c.ValToBin(v)
            self._populate_column_set_map(c, o, v, column_set_map)
        # compute the set of pids that's relevant to this query
        relevant_pids = set()
        first = True
        for cid in column_set_map:
            if first:
                relevant_pids = column_set_map[cid]
                first = False
            else:
                relevant_pids = relevant_pids.intersection(column_set_map[cid])
        total_card = 0
        # for each pid, check full or partial coverage
        # if full coverage, just add the full count
        # otherwise, estimate the partial sum with uniform spread assumption
        for pid in relevant_pids:
            total_card += self._estimate_cardinality_per_partition(
                self.partitions[pid], columns, operators, vals)
        self.OnEnd()
        return total_card

    def Size(self):
        total_size = 15 * 2 * 4
        for p in self.partitions:
            total_size += p.Size()
        total_size += 24 * (len(self.partitions) - 1)
        return total_size


'''
class Partition(object):
    def __init__(self):
        # a dictionary of {attribute: (low, high)}
        self.boundaries = {}
        # a list of row id that belongs to this partition
        self.data_points = []
        # per-column uniform spread
        self.uniform_spreads = []
        # per-distinct value density
        self.density = None
        self.col_value_list = {}
        self.rowid_to_position = {}

def create_partition(partitions):


def build_Multi_Histogram(table, max_limit):
    partitions = []
    if max_limit<2:
        p = Partition()
        for col in table.columns:
            p.boundaries[col.name] = (table.data[col.name].min(), table.data[col.name].max())


    if
    def create_first_partition(self, table):
        dslf

'''

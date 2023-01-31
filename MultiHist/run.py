from MultiHist.multi_hist import *
import pickle

# predictions bfp
bfp = pd.read_csv('data/Black_Friday/Black_Friday_Purchase_num.csv', escapechar='\\', encoding='utf-8',
                  low_memory=False, quotechar='"').iloc[:, 1:]
sql_csv_file = 'data/Black_Friday/black_friday_purchase_sql_test.csv'
limit = 5000
partitions = bulid_hist(bfp, limit)
pred_card, true_card = query_evaluate(sql_csv_file, partitions)
with open('MultiHist/bfp_all_multihist_5000.model','wb') as f:
    pickle.dump(partitions, f)
pd.DataFrame({'pred': pred_card, 'true': true_card}).to_csv('results/bfp_all_multi_hist_5000.csv', index=False, encoding='utf-8')


# fps
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
fps = pd.read_csv('data/gpu_game_fps/fps_num_lower.csv', escapechar='\\', encoding='utf-8',
                  low_memory=False, quotechar='"', usecols=usecols)
sql_csv_name = 'data/gpu_game_fps/fps_sql_test.csv'
limit = 5000
partitions = bulid_hist(fps, limit)
with open('MultiHist/fps_all_multihist_5000.model','rb') as f:
    partitions = pickle.load(f)
pred_card, true_card = query_evaluate(sql_csv_file, partitions)

n = 0
i = 0
for p in partitions:
    n = n + p.n_total
    if p.n_total>=80:
        print(i)
    i = i+1
print(n)

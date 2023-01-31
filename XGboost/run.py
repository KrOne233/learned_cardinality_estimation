import random

import XGboost.xgboost as xgboost

path_train = 'data/Black_Friday/black_friday_purchase_sql_train'
path_test = 'data/Black_Friday/black_friday_purchase_sql_test'
min_max_file = 'data/Black_Friday/column_min_max_vals.csv'
pattern2model = xgboost.train_for_all_pattern(path_train, min_max_file)
xgboost.test_for_all_pattern(path_test, 'bfp', pattern2model, min_max_file)


path_train = 'data/gpu_game_fps/fps_sql_train'
path_test = 'data/gpu_game_fps/fps_sql_test'
min_max_file = 'data/gpu_game_fps/column_min_max_vals.csv'
pattern2model = xgboost.train_for_all_pattern(path_train, min_max_file)
xgboost.test_for_all_pattern(path_test, 'fps', pattern2model, min_max_file)

i=1
while i<8:
    path_train = f'data/Black_Friday/2_col_sql/bfp_2col_{i}_train'
    path_test = f'data/Black_Friday/2_col_sql/bfp_2col_{i}_test'
    min_max_file = 'data/Black_Friday/column_min_max_vals.csv'
    pattern2model = xgboost.train_for_all_pattern(path_train, min_max_file)
    xgboost.test_for_all_pattern(path_test, f'bfp_2col_{i}', pattern2model, min_max_file)
    i = i+2




i = 1
while i<38:
    path_train = f'data/gpu_game_fps/2_col_sql/fps_2col_{i}_train'
    path_test = f'data/gpu_game_fps/2_col_sql/fps_2col_{i}_test'
    min_max_file = 'data/gpu_game_fps/column_min_max_vals.csv'
    pattern2model = xgboost.train_for_all_pattern(path_train, min_max_file)
    xgboost.test_for_all_pattern(path_test, f'fps_2col_{i}', pattern2model, min_max_file)
    i = i+4
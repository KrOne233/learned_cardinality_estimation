import XGboost.xgboost as xgboost

path_train = 'data/Black_Friday/black_friday_purchase_sql_train'
path_test = 'data/Black_Friday/black_friday_purchase_sql_test'
min_max_file = 'data/Black_Friday/column_min_max_vals.csv'
pattern2model = xgboost.train_for_all_pattern(path_train, min_max_file)
xgboost.test_for_all_pattern(path_test, 'black_friday_purchase', pattern2model, min_max_file)

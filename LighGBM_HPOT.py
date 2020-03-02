params = {
    # better accuracy
    'learning_rate':        hp.choice('learning_rate',      [0.0001, 0.001, 0.01]),
    'boosting_type':        hp.choice('boosting_type',      ['gbdt','dart']),
    'max_bin':              hp.choice('max_bin',            [200, 255, 300]),
    'num_leaves':           hp.choice('num_leaves',         [200, 300, 400]),

    # avoid overfit
    'min_data_in_leaf':     hp.choice('min_data_in_leaf',   [300, 500, 700]),
    'feature_fraction':     hp.choice('feature_fraction',   [0.6, 0.8, 0.9]),
    'bagging_fraction':     hp.choice('bagging_fraction',   [0.6, 0.8, 0.95]),
    'bagging_freq':         hp.choice('bagging_freq',       [2, 5, 8]),
    'min_gain_to_split':    hp.choice('min_gain_to_split',  [2, 5, 8]),
    'lambda_l1':            hp.choice('lambda_l1',          [0, 0.4, 0.8]),
    'lambda_l2':            hp.choice('lambda_l2',          [0, 0.5, 2]),

    # parameters won't change
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_error',
    'num_threads':10, # for the best speed, set this to the number of real CPU cores
}


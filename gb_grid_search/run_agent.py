"""Grid search to tune hyperparameters"""
import ast
import datetime
import sys

sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')

import pickle
import numpy as np
import lightgbm as lgb

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from logger import get_clean_logger

LOG = get_clean_logger(logger_name = Path(__file__).name)


def train():
    """Training function to call in our weights and biases grid search."""
    print(f'STARTING AT TIME {datetime.datetime.now()}')

    # Load in data
    LOG.info('Load data!!!')
    with open('/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/rf_cv_data.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)

    # Retrieve grid parameters
    agent_i = int(sys.argv[0])
    max_bins = int(sys.argv[1])
    num_iters = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_leaves = int(sys.argv(4))

    params = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class': 5,  # Specify the number of classes
        'metric': 'multi_logloss',
        'max_bin': max_bins,                # ***  [100 ---> 15000]
        'num_iterations': num_iters,        # ***  [400 ---> 1000]
        'learning_rate': learning_rate,     # ***  [0.1  ---> 0.0001]
        'num_leaves': num_leaves            # ***  [30 ---> 150]
    }


    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
    kf = StratifiedKFold(n_splits=10, random_state=22, shuffle=True)


    ######################## TRAIN ########################

    # Arrays to store true and pred data in
    all_true = []
    all_pred_proba = []

    # X and y set
    # TODO: IMPORT DATA !!!!!!!!


    for train_index, test_index in kf.split(X_train, y_train):

        ### NN-INFERRED DATA ###
        # Split the data into training and test sets for the current fold
        X_train_set, X_val = X_train[train_index], X_train[test_index]
        y_train_set, y_val = y_train[train_index], y_train[test_index]
        train_set = lgb.Dataset(X_train_set, label=y_train_set)

        # Fit and get confusion matrix
        # Non-balanced weights
        bst = lgb.train(params=params, train_set=train_set)
        y_pred_proba = bst.predict(X_val)

        # Get the info for the purity vs. completeness curve
        all_true.append(y_val)
        all_pred_proba.append(y_pred_proba)


    # Dump data into results file
    results = [all_true, all_pred_proba]
    with open(f'/n/home04/aboesky/berger/Weird_Galaxies/gb_grid_search/results/results_{agent_i}.pkl', 'wb') as f:
        pickle.dump((params, results), f)

    print(f'FINISHED AT TIME {datetime.datetime.now()}')


if __name__ == '__main__':
    train()

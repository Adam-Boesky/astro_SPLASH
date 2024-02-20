"""Grid search to tune hyperparameters"""
import ast
import datetime
import sys

sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')

import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from logger import get_clean_logger
from sklearn.ensemble import RandomForestClassifier

LOG = get_clean_logger(logger_name = Path(__file__).name)


def train():
    """Training function to call in our weights and biases grid search."""
    print(f'STARTING AT TIME {datetime.datetime.now()}')

    # Load in data
    LOG.info('Load data!!!')
    with open('/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/rf_cv_data.pkl', 'rb') as f:
        X, y = pickle.load(f)

    # Retrieve grid parameters
    try:
        weights = ast.literal_eval(sys.argv[-1])
    except:
        weights = str(sys.argv[-1])
    agent_i = int(sys.argv[-2])


    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
    kf = StratifiedKFold(n_splits=10, random_state=22, shuffle=True)
    rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=22, class_weight=weights)


    ######################## TRAIN ########################

    # Arrays to store true and pred data in
    all_true = []
    all_pred = []

    # X and y set
    pstar_X = X
    pstar_y = y

    # Iterate over each fold
    for train_index, test_index in kf.split(pstar_X, pstar_y):

        ### NN-INFERRED DATA ###
        # Split the data into training and test sets for the current fold
        X_train, X_test = pstar_X[train_index], pstar_X[test_index]
        y_train, y_test = pstar_y[train_index], pstar_y[test_index]

        # Fit and get confusion matrix
        # Non-balanced weights
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)

        # Append data
        all_true.append(y_test)
        all_pred.append(y_pred)


    # Dump data into results file
    results = [all_true, all_pred]
    with open(f'/n/home04/aboesky/berger/Weird_Galaxies/flat_rf_grid_search/results/results_{agent_i}.pkl', 'wb') as f:
        pickle.dump((weights, results), f)

    print(f'FINISHED AT TIME {datetime.datetime.now()}')


if __name__ == '__main__':
    train()

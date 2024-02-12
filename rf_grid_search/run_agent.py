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
from two_step_classifier import TwoStepClassifier

LOG = get_clean_logger(logger_name = Path(__file__).name)


def train():
    """Training function to call in our weights and biases grid search."""
    print(f'STARTING AT TIME {datetime.datetime.now()}')

    # Load in data
    LOG.info('Load data!!!')
    with open('/n/holystore01/LABS/berger_lab/Users/aboesky/Weird_Galaxies/rf_cv_data.pkl', 'rb') as f:
        X, y = pickle.load(f)

    # Retrieve grid parameters
    thresh = float(sys.argv[-1])
    try:
        cc_weight = ast.literal_eval(sys.argv[-2])
    except:
        cc_weight = str(sys.argv[-2])
    agent_i = int(sys.argv[-3])


    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
    kf = StratifiedKFold(n_splits=10, random_state=22, shuffle=True)
    two_step_classifier = TwoStepClassifier(ia_thresh=thresh)


    ######################## TRAIN ########################
    # K fold confusion matrix
    kf = StratifiedKFold(n_splits=5, random_state=22, shuffle=True)
    two_step_classifier = TwoStepClassifier(ia_thresh=thresh, cc_class_weights=cc_weight)

    # Initialize a matrix to hold the summed confusion matrix
    cumulative_cm = np.array([[0 for _ in range(5)] for _ in range(5)])

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
        two_step_classifier.fit(X_train, y_train)
        y_pred = two_step_classifier.predict(X_test)
        cumulative_cm += confusion_matrix(y_test, y_pred, labels=[i for i in range(5)])

    # Store the mean purity of CC classes
    cm_col_norms = np.sum(cumulative_cm, axis=0)
    cm_norm = cumulative_cm / cm_col_norms
    cm_norm[np.isnan(cm_norm)] = 0

    # Get mean purity
    mean_purity = np.mean(np.diag(cm_norm)[1:])

    with open(f'/n/home04/aboesky/berger/Weird_Galaxies/rf_grid_search/results/results_{agent_i}.pkl', 'wb') as f:
        params = [thresh, cc_weight]
        pickle.dump((params, mean_purity), f)
    print(f'FINISHED AT TIME {datetime.datetime.now()}')


if __name__ == '__main__':
    train()

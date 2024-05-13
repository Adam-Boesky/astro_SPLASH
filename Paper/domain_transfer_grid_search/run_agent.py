"""Grid search to tune hyperparameters"""
import ast
import datetime
import os
import shutil
import sys

sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from logger import get_clean_logger
from neural_net import (CustomLoss, checkpoint, get_model, get_tensor_batch, resume)
from sed_nn import load_and_preprocess

LOG = get_clean_logger(logger_name = Path(__file__).name)


def train():
    """Training function to call in our weights and biases grid search."""
    print(f'STARTING AT TIME {datetime.datetime.now()}')

    # Load in data
    LOG.info('Load data!!!')
    all_photo, photo_X_train, photo_y_train, photo_Xerr_train, \
            photo_yerr_train, photo_X_test, photo_y_test, photo_Xerr_test, \
            photo_yerr_test, photo_mean, photo_std, photo_err_norm = load_and_preprocess()

    # Retrieve grid parameters
    learning_rate = float(sys.argv[-1])
    num_linear_output_layers = int(sys.argv[-2])
    nodes_per_layer = ast.literal_eval(sys.argv[-3])
    batch_size = int(sys.argv[-4])
    agent_i = int(sys.argv[-5])

    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
    torch.set_default_dtype(torch.float64)
    model = get_model(num_inputs=5, num_outputs=13, nodes_per_layer=nodes_per_layer, num_linear_output_layers=num_linear_output_layers)
    loss_fn = CustomLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    ######################## TRAIN ########################
    # Training parameters
    n_epochs = 1000
    batch_size = int(batch_size)
    batches_per_epoch = int(len(photo_y_train) / batch_size)
    LOG.info('Batch Size = %i', batch_size)

    # Early stop stuff
    best_loss = 1E100
    best_epoch = -1

    # Training loop
    losses_per_epoch = {'train': [], 'test': []}

    # Temporary directory to store state in
    tmp_dir = TemporaryDirectory()
    best_model_path = os.path.join(tmp_dir.name, 'best_model.pkl')

    # Training loop
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(batches_per_epoch):

            # Get batch
            start = i * batch_size
            end = start + batch_size
            X_batch = get_tensor_batch(photo_X_train, start, end)
            y_batch = get_tensor_batch(photo_y_train, start, end)
            yerr_batch = get_tensor_batch(photo_yerr_train, start, end)

            # Predict and gradient descent
            model.train()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch, yerr_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = (epoch_loss / batches_per_epoch)
        model.eval()
        losses_per_epoch['train'].append(avg_train_loss)
        test_pred = model(torch.from_numpy(photo_X_test))
        test_loss = loss_fn(test_pred, torch.from_numpy(photo_y_test), torch.from_numpy(photo_yerr_test))
        losses_per_epoch['test'].append(test_loss.item())
        LOG.info('Epoch %i/%i finished with avg training loss = %.3f', epoch + 1, n_epochs, avg_train_loss)

        # Always store best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            checkpoint(model, best_model_path)

        # Early stopping
        elif epoch - best_epoch >= 50:
            LOG.info('Loss has not decreased in 50 epochs, early stopping. Best test loss is %.3f', best_loss)
            break

    # Load best model
    resume(model, best_model_path)
    LOG.info('!!!Finished Training!!!')
    tmp_dir.cleanup()

    with open(f'/n/home04/aboesky/berger/Weird_Galaxies/domain_transfer_grid_search/results/results_{agent_i}.pkl', 'wb') as f:
        params = [learning_rate, num_linear_output_layers, nodes_per_layer, batch_size]
        pickle.dump((params, losses_per_epoch), f)
    print(f'FINISHED AT TIME {datetime.datetime.now()}')


if __name__ == '__main__':
    train()

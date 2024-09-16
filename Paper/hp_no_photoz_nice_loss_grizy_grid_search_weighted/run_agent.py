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
from torch import nn

from logger import get_clean_logger
from neural_net import (CustomLoss, checkpoint, get_model, get_tensor_batch, resume)
from host_prop_nn import load_and_preprocess

LOG = get_clean_logger(logger_name = Path(__file__).name)

class CustomLossExpz(nn.Module):
    """A custom loss function. Basically the MSE but we divide by std"""
    def __init__(self):
        super(CustomLossExpz, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, target_err: torch.Tensor) -> torch.Tensor:
        # Copy the tensors to avoid in-place modification
        preds_transformed = preds.clone()
        targets_transformed = targets.clone()
        target_errs_transformed = target_err.clone()

        # Apply the transformation to the cloned tensors
        preds_transformed[:, 2] = 10 ** (preds[:, 2] * cat_std[2] + cat_mean[2])
        targets_transformed[:, 2] = 10 ** (targets[:, 2] * cat_std[2] + cat_mean[2])
        target_errs_transformed[:, 2] *= cat_std[2]
        target_errs_transformed[:, 2] = torch.abs(target_err[:, 2] * 2.302585092994046 * targets_transformed[:, 2])  # ln(10) = 2.302585092994046

        # Compute the loss using the transformed tensors
        loss = torch.mean(torch.div((preds_transformed - targets_transformed) ** 2, target_errs_transformed))

        return loss


class WeightedCustomExpZLoss(nn.Module):
    """A custom loss function. Basically the MSE but we divide by std"""
    def __init__(self, exponent: float):
        super(WeightedCustomExpZLoss, self).__init__()
        self.exponent = exponent

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, target_err: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        # Copy the tensors to avoid in-place modification
        preds_transformed = preds.clone()
        targets_transformed = targets.clone()
        target_errs_transformed = target_err.clone()

        # Apply the transformation to the cloned tensors
        preds_transformed[:, 2] = 10 ** (preds[:, 2] * cat_std[2] + cat_mean[2])
        targets_transformed[:, 2] = 10 ** (targets[:, 2] * cat_std[2] + cat_mean[2])
        target_errs_transformed[:, 2] *= cat_std[2]
        target_errs_transformed[:, 2] = torch.abs(target_err[:, 2] * 2.302585092994046 * targets_transformed[:, 2])  # ln(10) = 2.302585092994046

        # Compute the loss using the transformed tensors
        z_weight = (1 - target_z) ** (self.exponent) + 0.05
        return torch.mean( torch.div(z_weight * (preds_transformed - targets_transformed) * (preds_transformed - targets_transformed), target_errs_transformed) )


def train():
    """Training function to call in our weights and biases grid search."""
    print(f'STARTING AT TIME {datetime.datetime.now()}')

    # Load in data
    LOG.info('Load data!!!')
    global cat_mean
    global cat_std
    all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, \
            cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm = load_and_preprocess()
    photo_train = photo_train[:, :5]
    photo_test = photo_test[:, :5]

    # Retrieve grid parameters
    learning_rate = float(sys.argv[-1])
    num_linear_output_layers = int(sys.argv[-2])
    nodes_per_layer = ast.literal_eval(sys.argv[-3])
    batch_size = int(sys.argv[-4])
    agent_i = int(sys.argv[-5])

    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
    print('NPL', nodes_per_layer)
    torch.set_default_dtype(torch.float64)
    model = get_model(num_inputs=5, num_outputs=3, nodes_per_layer=nodes_per_layer, num_linear_output_layers=num_linear_output_layers)
    weight_exp = 6.0
    loss_fn = WeightedCustomExpZLoss(exponent=weight_exp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


    ######################## TRAIN ########################
    # Training parameters
    n_epochs = 2500
    batch_size = int(batch_size)
    batches_per_epoch = int(len(cat_train) / batch_size)
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
            photo_batch = get_tensor_batch(photo_train, start, end)
            cat_batch = get_tensor_batch(cat_train, start, end)
            cat_err_batch = get_tensor_batch(cat_err_train, start, end)

            # Predict and gradient descent
            model.train()
            cat_pred = model(photo_batch)
            loss = loss_fn(cat_pred, cat_batch, cat_err_batch, (cat_batch[:, -1] * cat_std[-1] + cat_mean[-1]).unsqueeze(1))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = (epoch_loss / batches_per_epoch)
        model.eval()
        losses_per_epoch['train'].append(avg_train_loss)
        test_pred = model(torch.from_numpy(photo_test))
        test_loss = loss_fn(test_pred, torch.from_numpy(cat_test), torch.from_numpy(cat_err_test), torch.from_numpy(cat_test[:, -1] * cat_std[-1] + cat_mean[-1]).unsqueeze(1))
        losses_per_epoch['test'].append(test_loss.item())
        LOG.info('Epoch %i/%i finished with avg training loss = %.3f', epoch + 1, n_epochs, avg_train_loss)

        # Always store best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            checkpoint(model, best_model_path)

        # Early stopping
        elif epoch - best_epoch >= 500:
            LOG.info('Loss has not decreased in 50 epochs, early stopping. Best test loss is %.3f', best_loss)
            break

    # Load best model
    resume(model, best_model_path)
    LOG.info('!!!Finished Training!!!')
    tmp_dir.cleanup()

    with open(f'/n/home04/aboesky/berger/Weird_Galaxies/hp_no_photoz_nice_loss_grizy_grid_search_weighted/results/results_{agent_i}.pkl', 'wb') as f:
        params = [learning_rate, num_linear_output_layers, nodes_per_layer, batch_size]
        pickle.dump((params, losses_per_epoch), f)
    print(f'FINISHED AT TIME {datetime.datetime.now()}')


if __name__ == '__main__':
    train()

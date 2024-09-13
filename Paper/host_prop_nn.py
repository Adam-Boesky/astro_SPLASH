import os
import pickle
import numpy as np
from pathlib import Path

import torch

from sklearn.model_selection import train_test_split
from logger import get_clean_logger
from torch import nn
from neural_net import (WeightedCustomLoss, CustomLoss, checkpoint, get_model, get_tensor_batch,
                        plot_real_v_preds, plot_training_loss, resume, normalize_arr)

LOG = get_clean_logger(logger_name = Path(__file__).name)  # Get my beautiful logger
VERBOSE = False                 # Whether logging should be verbose
CLUSTER = False                  # Whether we are on the cluster or not

# Parameters for skipping different parts of this file
SKIP_TRAINING = False
SKIP_PREPROCESSING = False
SKIP_PLOTTING = False

# Paths to different data
PATH_TO_DATA = '/Volumes/T7/ay_98_data/Zou_data'
PATH_TO_TRAINING_DATA = 'Volumes/T7/ay_98_data/pickled_data'
if CLUSTER:
    PATH_TO_CLEAN_DATA = '/n/holystore01/LABS/berger_lab/Users/aboesky/weird_galaxy_data'
else:
    PATH_TO_CLEAN_DATA = '/Users/adamboesky/Research/ay98/clean_data'


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

def load_and_preprocess():
    """Load and preprocess our data."""
    ######################## IMPORT DATA ########################
    LOG.info('Importing photometry data')
    with open(os.path.join(PATH_TO_CLEAN_DATA, 'all_photometry.pkl'), 'rb') as f:
        all_photo = pickle.load(f)
    photo = all_photo['data']
    photo_err = all_photo['data_err']
    photo = photo[:, :5]
    photo_err = photo_err[:, :5]

    # Take log of the fluxes to make the distributions better
    photo_err = np.abs(photo_err / (photo * np.log(10)))
    photo = np.log10(photo)

    with open(os.path.join(PATH_TO_CLEAN_DATA, 'all_cat.pkl'), 'rb') as f:
        all_cat = pickle.load(f)
    cat = all_cat['data']
    LOG.info('Fixing the error for %i objects', np.sum(all_cat['data_err'][:, 2] == 0.01))
    all_cat['data_err'][:, 2][all_cat['data_err'][:, 2] == 0.01] = 0.001 # Drop the spectroscopic errors down from the already low error

    # Drop photozs
    specz_mask = all_cat['data_err'][:, 2] == 0.001

    LOG.info('Dropping %i / %i photozs!', np.sum(~specz_mask), len(specz_mask))
    cat = cat[specz_mask]
    cat_err = all_cat['data_err'][specz_mask]
    photo = photo[specz_mask]
    photo_err = photo_err[specz_mask]

    LOG.info('Importing photometry data')
    # Filter out z>1
    z_local_mask = cat[:, 2] <= 1
    # add additional filter for potentially bad bands
    z_local_mask &= cat[:, 2] > 0.0126  # this corresponds to a pileup of tiny z in the photoz fits
    photo = photo[z_local_mask]
    photo_err = photo_err[z_local_mask]
    cat = cat[z_local_mask]
    cat_err = cat_err[z_local_mask]

    # Drop bad bands (ch1, mips, pacs100)
    # good_bands = [i for i, t in enumerate(all_photo['sorted_filters']) if t not in ['CH1', 'MIPS24', 'MIPS70', 'PACS100']]
    # all_photo['sorted_filters'] = [all_photo['sorted_filters'][i] for i in good_bands]
    # all_photo['sorted_wavelengths'] = [all_photo['sorted_wavelengths'][i] for i in good_bands]
    # photo = photo[:, good_bands]
    # photo_err = photo_err[:, good_bands]

    # Log transform the redshifts
    cat_err[:, 2] = np.abs(cat_err[:, 2] / (np.log(10) * cat[:, 2]))
    cat[:, 2] = np.log10(cat[:, 2])

    ######################## PRE PROCESSING ########################
    # Filter out nans
    nan_mask = np.isnan(cat).any(axis=1)
    photo_norm, photo_mean, photo_std, photo_err_norm = normalize_arr(photo[~nan_mask], errors=photo_err[~nan_mask])
    global cat_mean
    global cat_std
    cat_norm, cat_mean, cat_std, cat_err_norm = normalize_arr(cat[~nan_mask], errors=cat_err[~nan_mask])
    print('HEEERRRREE', cat_norm)
    print(cat_mean)
    LOG.info('Photo stats:\n \tmean = %s\n \tstd = %s', photo_mean, photo_std)
    LOG.info('Catalog stats:\n \tmean = %s\n \tstd = %s', cat_mean, cat_std)
    LOG.info('Length = %i', len(photo))

    # Split train and test sets
    photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, cat_err_test = \
        train_test_split(photo_norm, cat_norm, photo_err_norm, cat_err_norm, shuffle=True, test_size=0.2, random_state=22)


    return all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm


def train_and_store_nn():

    ######################## IMPORT DATA ########################
    if not SKIP_PREPROCESSING:
        all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, \
            cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm = load_and_preprocess()






    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################

    # Training parameters
    n_epochs = 10_000
    # # V1: [4096, [18, 15, 12, 9, 6, 4], 3, 0.01]
    # batch_size = 4096
    # nodes_per_layer = [18, 15, 12, 9, 6, 4]
    # num_linear_output_layers = 3
    # learning_rate = 0.01
    # loss_fn = WeightedCustomLoss()
    # # V2 GOOD BANDS
    # nodes_per_layer = [12, 10, 8, 6, 4]
    # num_linear_output_layers = 3
    # learning_rate = 0.001
    # batch_size = 1024
    # loss_fn = CustomLoss()
    # TUNED WEIGHTED: [0.01, 3, [18, 15, 12, 9, 6, 4], 4096, 8.0]
    # batch_size = 4096
    # nodes_per_layer = [18, 15, 12, 9, 6, 4]
    # num_linear_output_layers = 3
    # learning_rate = 0.01
    # weight_exp = 6.0
    # loss_fn = WeightedCustomLoss(exponent=weight_exp)
    # # SPECZ ONLY: [4096, [18, 13, 8, 4], 3, 0.01]
    # nodes_per_layer = [18, 13, 8, 4]
    # num_linear_output_layers = 3
    # learning_rate = 0.01
    # batch_size = 4096
    # SPECZ ONLY AND REDSHIFT LOSS:   [2048, [18, 15, 12, 9, 6, 4], 3, 0.01]
    # n_epochs = 1000
    # nodes_per_layer = [18, 15, 12, 9, 6, 4]
    # num_linear_output_layers = 3
    # learning_rate = 0.01
    # batch_size = 2048
    # SPECZ ONLY, AND GRIZY: [512, [5, 5, 4], 3, 0.01]
    nodes_per_layer = [5, 5, 4]
    num_linear_output_layers = 3
    learning_rate = 0.01
    batch_size = 512

    loss_fn = CustomLossExpz()
    torch.set_default_dtype(torch.float64)
    model = get_model(num_inputs=5, num_outputs=3, nodes_per_layer=nodes_per_layer, num_linear_output_layers=num_linear_output_layers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)





    ######################## TRAIN ########################
    if not SKIP_TRAINING:
        batches_per_epoch = int(len(cat_train) / batch_size)
        LOG.info('Batch Size = %i', batch_size)

        # Early stop stuff
        best_loss = 1E100
        best_epoch = -1

        # Training loop
        losses_per_epoch = {'train': [], 'test': []}
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
                loss = loss_fn(cat_pred, cat_batch, cat_err_batch)# , (cat_batch[:, -1] * cat_std[-1] + cat_mean[-1]).unsqueeze(1))
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_hash = int(40 * i / batches_per_epoch)
                if VERBOSE:
                    LOG.info('Batch %i/%i |' +  num_hash * '#' + ' ' * (40 - num_hash) + '|', i + 1, batches_per_epoch)
            avg_train_loss = (epoch_loss / batches_per_epoch)
            model.eval()
            losses_per_epoch['train'].append(avg_train_loss)
            test_pred = model(torch.from_numpy(photo_test))
            test_loss = loss_fn(test_pred, torch.from_numpy(cat_test), torch.from_numpy(cat_err_test))# , torch.from_numpy(cat_test[:, -1] * cat_std[-1] + cat_mean[-1]).unsqueeze(1))
            losses_per_epoch['test'].append(test_loss.item())
            LOG.info('Epoch %i/%i finished with avg training loss = %.3f', epoch + 1, n_epochs, avg_train_loss)

            # Always store best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                # checkpoint(model, "/Users/adamboesky/Research/ay98/Weird_Galaxies/host_prop_no_photozs.pkl")
                checkpoint(model, "/Users/adamboesky/Research/ay98/Weird_Galaxies/host_prop_no_photozs_nice_loss.pkl")

            # Early stopping
            elif epoch - best_epoch >= 1000:
                LOG.info('Loss has not decreased in 50 epochs, early stopping. Best test loss is %.3f', best_loss)
                break

            # Decrease learning rate on 1500th epoch
            if epoch == 1499:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5

        # Plot training performance
        with open('final_host_prop_train_history.pkl', 'wb') as f:
            pickle.dump(losses_per_epoch, f)
        plot_training_loss(losses_per_epoch['train'], test_losses=losses_per_epoch['test'], filename='/Users/adamboesky/Research/ay98/Weird_Galaxies/Paper/V2_host_prop_nn_training_plots/final_loss_v_epoch.png')

    # Load best model
    # resume(model, '/Users/adamboesky/Research/ay98/Weird_Galaxies/host_prop_no_photozs.pkl')
    resume(model, '/Users/adamboesky/Research/ay98/Weird_Galaxies/host_prop_no_photozs_nice_loss.pkl')


    ######################## CHECK RESULTS AND STORE MODEL ########################
    if not SKIP_PLOTTING:
        LOG.info('Plotting')
        if SKIP_TRAINING:
            with open(os.path.join(PATH_TO_CLEAN_DATA, 'most_recent_data.pkl'), 'rb') as f:
                (all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm) = pickle.load(f)

        model.eval()
        test_pred: torch.Tensor = model(torch.from_numpy(photo_test))
        test_pred_untrans = test_pred.detach().numpy()
        for idx in range(test_pred.shape[1]):
            plot_real_v_preds(cat_test[:, idx] * cat_std[idx] + cat_mean[idx], test_pred_untrans[:, idx] * cat_std[idx] + cat_mean[idx], real_err=cat_err_test[:, idx] * cat_std[idx], param=all_cat['keys'][idx], plot_dirname='Paper/V2_host_prop_nn_training_plots', filename_postfix=all_cat['keys'][idx])


if __name__ == '__main__':
    train_and_store_nn()

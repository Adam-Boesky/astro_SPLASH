import os
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from logger import get_clean_logger
from neural_net import (CustomLoss, checkpoint, get_model, get_tensor_batch,
                        normalize_arr, plot_real_v_preds, plot_training_loss,
                        resume)

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


def load_and_preprocess():
    """Load and preprocess our data."""
    ######################## IMPORT DATA ########################
    LOG.info('Importing photometry data')
    with open(os.path.join(PATH_TO_CLEAN_DATA, 'all_photometry.pkl'), 'rb') as f:
        all_photo = pickle.load(f)
    photo = all_photo['data']
    photo_err = all_photo['data_err']

    # Take log of the fluxes to make the distributions better
    photo_err = np.abs(photo_err / (photo * np.log(10)))
    photo = np.log10(photo)

    with open(os.path.join(PATH_TO_CLEAN_DATA, 'all_cat.pkl'), 'rb') as f:
        all_cat = pickle.load(f)
    cat = all_cat['data']
    LOG.info('Fixing the error for %i objects', np.sum(all_cat['data_err'][:, 2] == 0.01))
    all_cat['data_err'][:, 2][all_cat['data_err'][:, 2] == 0.01] = 0.001 # Drop the spectroscopic errors down from the already low error

    LOG.info('Importing photometry data')
    # Filter out z>1
    z_local_mask = cat[:, 2] <= 1
    photo = photo[z_local_mask]
    photo_err = photo_err[z_local_mask]
    cat = cat[z_local_mask]

    # Drop bad bands (ch1, mips, pacs100)
    good_bands = [i for i, t in enumerate(all_photo['sorted_filters']) if t not in ['CH1', 'MIPS24', 'MIPS70', 'PACS100']]
    all_photo['sorted_filters'] = [all_photo['sorted_filters'][i] for i in good_bands]
    all_photo['sorted_wavelengths'] = [all_photo['sorted_wavelengths'][i] for i in good_bands]
    photo = photo[:, good_bands]
    photo_err = photo_err[:, good_bands]


    ######################## PRE PROCESSING ########################
    # Filter out nans
    nan_mask = np.isnan(cat).any(axis=1)
    photo_norm, photo_mean, photo_std, photo_err_norm = normalize_arr(photo[~nan_mask], errors=photo_err[~nan_mask])
    cat_norm, cat_mean, cat_std = normalize_arr(cat[~nan_mask])
    print('HEEERRRREE', cat_norm)
    print(cat_mean)
    LOG.info('Photo stats:\n \tmean = %s\n \tstd = %s', photo_mean, photo_std)
    LOG.info('Catalog stats:\n \tmean = %s\n \tstd = %s', cat_mean, cat_std)
    LOG.info('Length = %i', len(photo))

    # Deal with upper limit fluxes
    for j in range(photo_err_norm.shape[1]):
        mask = photo_err_norm[:, j] == 0  # mask for upper limit
        photo_err_norm[:, j][mask] = np.abs(photo_norm[:, j][mask])  # set the error equal to the photo upper limit
        photo_norm[:, j][mask] = 0.01  # set the real value to some arbitrarily small value

    # Split train and test sets
    photo_train, photo_test, photo_err_train, photo_err_test = \
        train_test_split(photo_norm, photo_err_norm, shuffle=True, test_size=0.2, random_state=22)

    # Split into X and y
    LOG.info('Splitting into:\n\tX = %s\n\ty = %s', all_photo['sorted_filters'][:5], all_photo['sorted_filters'][5:])
    photo_X_train, photo_y_train, photo_Xerr_train, photo_yerr_train  = photo_train[:, :5], photo_train[:, 5:], photo_err_train[:, :5], photo_err_train[:, 5:]
    photo_X_test, photo_y_test, photo_Xerr_test, photo_yerr_test = photo_test[:, :5], photo_test[:, 5:], photo_err_test[:, :5], photo_err_test[:, 5:]

    return all_photo, photo_X_train, photo_y_train, photo_Xerr_train, photo_yerr_train, photo_X_test, photo_y_test, photo_Xerr_test, photo_yerr_test, photo_mean, photo_std, photo_err_norm


def train_and_store_nn():

    ######################## IMPORT DATA ########################
    if not SKIP_PREPROCESSING:
        all_photo, photo_X_train, photo_y_train, photo_Xerr_train, \
            photo_yerr_train, photo_X_test, photo_y_test, photo_Xerr_test, \
            photo_yerr_test, photo_mean, photo_std, photo_err_norm = load_and_preprocess()






    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################

    # Training parameters
    n_epochs = 1000
    nodes_per_layer = [6, 7, 7, 8]
    num_linear_output_layers = 2
    learning_rate = 9.999999999999999e-05
    batch_size = 256
    torch.set_default_dtype(torch.float64)
    model = get_model(num_inputs=5, num_outputs=9, nodes_per_layer=nodes_per_layer, num_linear_output_layers=num_linear_output_layers)
    loss_fn = CustomLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)





    ######################## TRAIN ########################
    if not SKIP_TRAINING:
        batches_per_epoch = int(len(photo_y_train) / batch_size)
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

                num_hash = int(40 * i / batches_per_epoch)
                if VERBOSE:
                    LOG.info('Batch %i/%i |' +  num_hash * '#' + ' ' * (40 - num_hash) + '|', i + 1, batches_per_epoch)
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
                checkpoint(model, "/Users/adamboesky/Research/ay98/Weird_Galaxies/V2_best_sed_model.pkl")

            # Early stopping
            elif epoch - best_epoch >= 50:
                LOG.info('Loss has not decreased in 50 epochs, early stopping. Best test loss is %.3f', best_loss)
                break

        # Plot training performance
        plot_training_loss(losses_per_epoch['train'], test_losses=losses_per_epoch['test'], filename='/Users/adamboesky/Research/ay98/Weird_Galaxies/V2_sed_nn_training_plots/loss_v_epoch.png')

    # Load best model
    resume(model, '/Users/adamboesky/Research/ay98/Weird_Galaxies/V2_best_sed_model.pkl')


    ######################## CHECK RESULTS AND STORE MODEL ########################
    if not SKIP_PLOTTING:
        LOG.info('Plotting')
        if SKIP_PREPROCESSING:
            with open(os.path.join(PATH_TO_CLEAN_DATA, 'most_recent_data.pkl'), 'rb') as f:
                (all_photo, photo_X_train, photo_y_train, photo_Xerr_train, \
                    photo_yerr_train, photo_X_test, photo_y_test, photo_Xerr_test, \
                    photo_yerr_test, photo_mean, photo_std, photo_err_norm) = pickle.load(f)

        model.eval()
        test_pred: torch.Tensor = model(torch.from_numpy(photo_X_test))
        test_pred_untrans = test_pred.detach().numpy()
        for idx in range(test_pred.shape[1]):
            up_lim_mask = photo_y_test == 0.01
            photo_y_test[up_lim_mask] = photo_yerr_test[up_lim_mask]  # set the measurements back to upper limits for plotting
            plot_real_v_preds(photo_y_test[:, idx] * photo_std[idx] + photo_mean[idx], test_pred_untrans[:, idx] * photo_std[idx] + photo_mean[idx], real_err=photo_yerr_test[:, idx] * photo_std[idx], param=f"log10({all_photo['sorted_filters'][5:][idx]}", plot_dirname='V2_sed_nn_training_plots', filename_postfix=all_photo['sorted_filters'][5:][idx])


if __name__ == '__main__':
    train_and_store_nn()

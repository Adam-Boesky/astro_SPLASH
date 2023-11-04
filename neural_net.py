import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from pathlib import Path
from logger import get_clean_logger
from typing import List, Union, Callable, Optional
from torch import nn
from astropy.io import fits
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer

LOG = get_clean_logger(logger_name = Path(__file__).name)  # Get my beautiful logger
VERBOSE = False                 # Whether logging should be verbose
CLUSTER = True                  # Whether we are on the cluster or not

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

torch.manual_seed(22)


def check_same_order(arr1: np.ndarray, arr2: np.ndarray, key: str = 'Tractor_ID'):
    """Check that two numpy arrays are ordered in the same way based on some ID value."""
    if len(arr1) != len(arr2):
        raise Exception(f"Data arrays are different lengths: {len(arr1)} != {len(arr2)}")
    if not np.all(arr1[key].astype(int) == arr2[key].astype(int)):
        raise Exception("Data arrays are not the same order!")
    pass


def normalize_arr(arr: np.ndarray, errors: np.ndarray = None, axis: int = 0) -> (float, float, float, Optional[np.ndarray]):
    """Normalize numpy array along given axis and its errors if given."""
    mean = np.mean(arr, axis=axis, keepdims=True)
    std = np.std(arr, axis=axis, keepdims=True)
    if not errors is None:
        return (arr - mean) / std, mean[0], std[0], errors / std
    else:
        return (arr - mean) / std, mean, std


def impute(data: np.ndarray) -> np.ndarray:
    imputer = KNNImputer(n_neighbors=5)
    # imputer.set_params(keep_empty_features=True)
    return imputer.fit_transform(data)


def get_tensor_batch(arr1: np.ndarray, start: int, stop: int) -> torch.Tensor:
    """Get a tensor batch from a given array given a start and stop"""
    return torch.from_numpy(arr1[start:stop])


def get_model(num_inputs: int, num_outputs: int, nodes_per_layer: List[int], num_linear_output_layers: int = 2) -> nn.Sequential:
    """Create a NN with given structure."""
    # Create model and add input layer
    model = nn.Sequential()
    model.add_module('input', nn.Linear(num_inputs, nodes_per_layer[0]))
    model.add_module(f'act_input', nn.ReLU())

    # Add hidden layers
    for i, nodes in enumerate(nodes_per_layer[:-1]):
        model.add_module(f'layer_{i}', nn.Linear(nodes, nodes_per_layer[i + 1]))
        model.add_module(f'act_{i}', nn.ReLU())

    # Add linear layers before the output to allow results to spread out after RuLU
    for i in range(num_linear_output_layers - 1):
        model.add_module(f'pre_output{i}', nn.Linear(nodes_per_layer[-1], nodes_per_layer[-1]))

    # Output layer
    model.add_module('output', nn.Linear(nodes_per_layer[-1], num_outputs))
    return model


def checkpoint(model, filepath):
    """Save pytorch model."""
    torch.save(model.state_dict(), filepath)


def resume(model, filepath):
    """Resume pytorch model."""
    model.load_state_dict(torch.load(filepath))


def plot_training_loss(losses: Union[list, np.ndarray], test_losses: np.ndarray = None, filename: str = 'loss_v_epoch.png'):
    """Plots the training loss as function of epoch"""
    # All epochs
    plt.figure(figsize=(10,5))
    epochs = np.arange(0, len(losses), 1) + 1

    plt.plot(epochs, np.log10(losses), label='Train')
    if test_losses is not None:
        plt.plot(epochs, np.log10(test_losses), label='Test')
        plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('log(Loss)')

    if isinstance(filename, str):
        plt.savefig(filename)
    
    # Plot from epoch 50 to the end
    plt.figure(figsize=(10,5))
    epochs = np.arange(50, len(losses), 1) + 1

    plt.plot(epochs, np.log10(losses[50:]), label='Train')
    if test_losses is not None:
        plt.plot(epochs, np.log10(test_losses[50:]), label='Test')
        plt.legend()

    plt.xlabel('Epoch')
    plt.ylabel('log(Loss)')
    
    if isinstance(filename, str):
        fname_list = filename.split('/')
        fname_list[-1] = 'ignore_start_' + fname_list[-1]
        plt.savefig('/'.join(fname_list))


def plot_real_v_preds(real: Union[list, np.ndarray, torch.Tensor], pred: Union[list, np.ndarray], param: str, real_err: Optional[Union[torch.Tensor, np.ndarray]] = None, pred_err: Optional[Union[torch.Tensor, np.ndarray]] = None, filename_postfix: str = ''):
    """Plots the real value versus the predicted value of a given parameter"""
    plt.figure(figsize=(10,5))
    if isinstance(real, torch.Tensor):
        real = real.detach().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().numpy()
    if isinstance(real_err, torch.Tensor):
        real_err = real_err.detach().numpy()
    if isinstance(pred_err, torch.Tensor):
        pred_err = pred_err.detach().numpy()
    if real_err is None:
        real_err = np.zeros((len(real)))
    else:
        real_err = real_err.flatten()
    if pred_err is None:
        pred_err = np.zeros((len(real)))
    else:
        pred_err = pred_err.flatten()
    if isinstance(real, np.ndarray):
        real = real.flatten()
    if isinstance(pred, np.ndarray):
        pred = pred.flatten()

    # Real v pred scatter
    _, caps, bars = plt.errorbar(real.flatten(), pred.flatten(), xerr=real_err, yerr=pred_err, ls = "None", color = "gray")
    [bar.set_alpha(0.01) for bar in bars]
    [cap.set_alpha(0.01) for cap in caps]
    plt.scatter(real, pred, s=3, alpha=0.2)
    plt.xlabel(fr'Real {param}')
    plt.ylabel(fr'Predicted {param}')
    center = np.mean(real)
    dist_width = np.max(real) - np.min(real)
    plt.axline((center, center), slope=1, color='black', linewidth=0.5)
    plt.xlim((center - dist_width*0.75, center + dist_width*1.25))
    scatter_xlims = plt.xlim()
    scatter_ylims = plt.ylim()
    plt.savefig(f'/n/home04/aboesky/berger/Weird_Galaxies/real_v_pred_scatter_{filename_postfix}.png')

    # Real v pred heatmaps
    heatmap, xedges, yedges = np.histogram2d(real, pred, range=[scatter_xlims, scatter_ylims], bins=(100, 100))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    plt.figure(figsize=(10,5))
    plt.imshow(np.log10(heatmap).T, extent=extent, origin='lower', aspect='auto', cmap='Blues')
    plt.axline((center, center), slope=1, color='black', linewidth=0.5)
    plt.colorbar(label="Log(Count)")
    plt.xlabel(fr'Real {param}')
    plt.ylabel(fr'Predicted {param}')
    plt.savefig(f'/n/home04/aboesky/berger/Weird_Galaxies/real_v_pred_heatmap_{filename_postfix}.png')

    # Real v pred histograms
    plt.figure(figsize=(10,5))
    plt.hist(real, bins=50, alpha=0.5, label='Real', density=True)
    plt.hist(pred, bins=50, alpha=0.5, label='Prediction', density=True)
    plt.ylabel('Frequency')
    plt.xlabel(param)
    plt.legend()
    plt.savefig(f'/n/home04/aboesky/berger/Weird_Galaxies/real_v_pred_hist_{filename_postfix}.png')

    # Fractional error histogram
    plt.figure(figsize=(10,5))
    frac_err = np.abs((pred.flatten() - real.flatten()) / real.flatten())
    frac_err_best_99p = frac_err[np.argsort(frac_err)[:int(0.99 * len(frac_err))]] # Take the best 99% of the frac err
    plt.hist(np.log10(frac_err_best_99p), bins=50, density=True)  # TODO: FIX ISSUES
    plt.xlabel(f'Top 99% of Log Fractional Error of {filename_postfix}')
    plt.ylabel('Frequency')
    # ind = np.argpartition(frac_err, -10)[-10:]
    # LOG.info('max val is %s', frac_err[ind])
    plt.savefig(f'/n/home04/aboesky/berger/Weird_Galaxies/frac_err_{filename_postfix}.png')


def get_np_data_from_fits(filepath: Union[str, Path], columns: list, transforms: List[Union[None, Callable]], vector_key: Optional[str] = None):
    """Get given fields in a numpy array from fits file path. Apply given transforms to the data. Return a vector of a column if given."""
    fits_data = np.array(fits.open(os.path.join(PATH_TO_DATA, filepath))[1].data)
    LOG.info('Importing %s', filepath)

    # Drop any severely Nan arrays (>50%)
    n = len(fits_data)
    good_columns = []
    good_transforms = []
    for col, trans in zip(columns, transforms):
        num_nans = np.count_nonzero(~np.isnan(fits_data[col]))
        if num_nans > 0.5 * n:
            LOG.info('Dropping column %s with %i / %i = %.2f%% NaNs', col, num_nans, n, (num_nans / n) * 100)
        else:
            LOG.info('Keeping column %s with %i / %i = %.2f%% NaNs', col, num_nans, n, (num_nans / n) * 100)
            good_columns.append(col)
            good_transforms.append(trans)

    # Transform the data
    out_arr = np.array([fits_data[col] for col in good_columns]).T
    out_arr = impute(out_arr)
    for idx, trans in enumerate(good_transforms):
        if trans is not None:
            out_arr[:,idx] = trans(out_arr[:,idx])
    LOG.info('Done imputing!')

    # If given vector key, return vector as well as data
    if vector_key is not None:
        return out_arr, fits_data[[vector_key]]
    return out_arr


class CustomLoss(nn.Module):
    """A custom loss function. Basically the MSE but we divide by std"""
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor, target_err: torch.Tensor) -> torch.Tensor:
        return torch.mean( torch.div((preds - targets) * (preds - targets), target_err) )


def ab_mag_to_flux(AB_mag: np.ndarray) -> np.ndarray:
    """Convert AB magnitude to flux"""
    return np.exp((AB_mag - 8.9) / -2.5) / 1000


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
    cat_err = all_cat['data_err']

    LOG.info('Importing photometry data')
    # Filter out z>1
    z_local_mask = cat[:, 2] <= 1
    photo = photo[z_local_mask]
    photo_err = photo_err[z_local_mask]
    cat = cat[z_local_mask]
    cat_err = cat_err[z_local_mask]


    ######################## PRE PROCESSING ########################
    # Filter out nans
    nan_mask = np.isnan(cat).any(axis=1)
    photo_norm, photo_mean, photo_std, photo_err_norm = normalize_arr(photo[~nan_mask], errors=photo_err[~nan_mask])
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
    # pickle_path = os.path.join(PATH_TO_TRAINING_DATA, 'es1_data.pkl')

    # if os.path.exists(pickle_path):
    #     with open(pickle_path, 'rb') as f:
    #         photo, photo_err, cat, cat_err = pickle.load(f)
    # else:
    # in_keys = ['f_MIPS24', 'f_MIPS70', 'f_MIPS160', 'f_PACS100', 'f_PACS160', 'f_SPIRE250', 'f_SPIRE350', 'f_SPIRE500']
    # in_err_keys = [f'ferr{in_k[1:]}' for in_k in in_keys]
    # in_transforms = [None for _ in range(len(in_keys))]
    # out_err_keys = ['Mstar_best_err']
    # out_keys = ['Mstar_best']
    # out_transforms = [lambda x: np.log(x)]
    # out_err_tranforms = [lambda x: np.abs(x / np.exp(cat).flatten())]

    # photo, photo_ids = get_np_data_from_fits(os.path.join(PATH_TO_DATA, 'photometry/es1_photcat.v1.fits'), in_keys, transforms=in_transforms, vector_key='Tractor_ID')
    # photo_err = get_np_data_from_fits(os.path.join(PATH_TO_DATA, 'photometry/es1_photcat.v1.fits'), in_err_keys, transforms=[None for _ in range(len(in_keys))])
    # cat, cat_ids = get_np_data_from_fits(os.path.join(PATH_TO_DATA, 'sed_catalog/es1.v1.fits'), out_keys, transforms=out_transforms, vector_key='Tractor_ID')
    # cat_err = get_np_data_from_fits(os.path.join(PATH_TO_DATA, 'sed_catalog/es1.v1.fits'), out_err_keys, transforms=out_err_tranforms)
    # check_same_order(cat_ids, photo_ids)  # Confirm that the object rows correspond to eachother
    if not SKIP_PREPROCESSING:
        all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, \
            cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm = load_and_preprocess()
        # LOG.info('Importing photometry data')
        # with open(f'/Users/adamboesky/Research/ay98/preprocessing/clean_data/all_photometry.pkl', 'rb') as f:
        #     all_photo = pickle.load(f)
        # photo = all_photo['data']
        # photo_err = all_photo['data_err']

        # # Take log of the fluxes to make the distributions better
        # photo_err = np.abs(photo_err / (photo * np.log(10)))
        # photo = np.log10(photo)

        # with open(f'/Users/adamboesky/Research/ay98/preprocessing/clean_data/all_cat.pkl', 'rb') as f:
        #     all_cat = pickle.load(f)
        # cat = all_cat['data']
        # LOG.info('Fixing the error for %i objects', np.sum(all_cat['data_err'][:, 2] == 0.01))
        # all_cat['data_err'][:, 2][all_cat['data_err'][:, 2] == 0.01] = 0.001 # Drop the spectroscopic errors down from the already low error
        # cat_err = all_cat['data_err']

        # # Filter out z>1
        # z_local_mask = cat[:, 2] <= 1
        # photo = photo[z_local_mask]
        # photo_err = photo_err[z_local_mask]
        # cat = cat[z_local_mask]
        # cat_err = cat_err[z_local_mask]






        # ######################## PRE PROCESSING ########################
        # nan_mask = np.isnan(cat).any(axis=1)
        # photo_norm, photo_mean, photo_std, photo_err_norm = normalize_arr(photo[~nan_mask], errors=photo_err[~nan_mask])
        # cat_norm, cat_mean, cat_std, cat_err_norm = normalize_arr(cat[~nan_mask], errors=cat_err[~nan_mask])
        # print('HEEERRRREE', cat_norm)
        # print(cat_mean)
        # LOG.info('Photo stats:\n \tmean = %s\n \tstd = %s', photo_mean, photo_std)
        # LOG.info('Catalog stats:\n \tmean = %s\n \tstd = %s', cat_mean, cat_std)
        # LOG.info('Length = %i', len(photo))

        # # Split train and test sets
        # photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, cat_err_test = \
        #     train_test_split(photo_norm, cat_norm, photo_err_norm, cat_err_norm, shuffle=True, test_size=0.2, random_state=22)
        
        # with open('/Users/adamboesky/Research/ay98/preprocessing/clean_data/most_recent_data.pkl', 'wb') as f:
        #     dat = (all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm)
        #     pickle.dump(dat, f)





    ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
    # According to results.ipynb the best hyperparams are:
    # batch_size, nodes_per_layer, num_linear_output_layers, learning_rate
    # [4096, [18, 15, 12, 9, 6, 4], 3, 0.01]

    # Training parameters
    n_epochs = 1000
    nodes_per_layer = [18, 15, 12, 9, 6, 4]
    num_linear_output_layers = 3
    learning_rate = 0.01
    batch_size = 4096
    torch.set_default_dtype(torch.float64)
    model = get_model(num_inputs=18, num_outputs=3, nodes_per_layer=nodes_per_layer, num_linear_output_layers=num_linear_output_layers)#BEST---nodes_per_layer=[18,12,7,4]) # without z limit [18,13,10,7,4]
    loss_fn = CustomLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # step_scheduler = MultiStepLR(optimizer, milestones=[200, 300, 350], gamma=0.1)





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
                loss = loss_fn(cat_pred, cat_batch, cat_err_batch)
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
            test_loss = loss_fn(test_pred, torch.from_numpy(cat_test), torch.from_numpy(cat_err_test))
            losses_per_epoch['test'].append(test_loss.item())
            LOG.info('Epoch %i/%i finished with avg training loss = %.3f', epoch + 1, n_epochs, avg_train_loss)

            # Always store best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                checkpoint(model, "/n/home04/aboesky/berger/Weird_Galaxies/best_model.pkl")

            # Early stopping
            elif epoch - best_epoch >= 50:
                LOG.info('Loss has not decreased in 50 epochs, early stopping. Best test loss is %.3f', best_loss)
                break

        # Plot training performance
        plot_training_loss(losses_per_epoch['train'], test_losses=losses_per_epoch['test'], filename='/n/home04/aboesky/berger/Weird_Galaxies/loss_v_epoch.png')

    # Load best model
    resume(model, '/n/home04/aboesky/berger/Weird_Galaxies/best_model.pkl')


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
            plot_real_v_preds(cat_test[:, idx] * cat_std[idx] + cat_mean[idx], test_pred_untrans[:, idx] * cat_std[idx] + cat_mean[idx], real_err=cat_err_test[:, idx] * cat_std[idx], param=all_cat['keys'][idx], filename_postfix=all_cat['keys'][idx])


if __name__ == '__main__':
    train_and_store_nn()

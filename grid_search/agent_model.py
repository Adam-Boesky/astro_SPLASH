"""Grid search to tune hyperparameters"""
import os
import sys

sys.path.append('/n/home04/aboesky/berger/Weird_Galaxies')

import wandb
import torch

from pathlib import Path
from tempfile import TemporaryDirectory

from logger import get_clean_logger
from neural_net import load_and_preprocess, get_model, get_tensor_batch, checkpoint, resume, CustomLoss

LOG = get_clean_logger(logger_name = Path(__file__).name)


def train(config=None):
    """Training function to call in our weights and biases grid search."""
    # Load in data
    all_cat, all_photo, photo_train, photo_test, cat_train, cat_test, photo_err_train, photo_err_test, cat_err_train, \
            cat_err_test, photo_norm, photo_mean, photo_std, photo_err_norm, cat_norm, cat_mean, cat_std, cat_err_norm = load_and_preprocess()
    
    with wandb.init(config=config):
        config = wandb.config

        ######################## MAKE NN, LOSS FN, AND OPTIMIZER  ########################
        torch.set_default_dtype(torch.float64)
        model = get_model(num_inputs=18, num_outputs=3, nodes_per_layer=config.nodes_per_layer, num_linear_output_layers=config.num_linear_output_layers)
        loss_fn = CustomLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


        ######################## TRAIN ########################
        # Training parameters
        n_epochs = config.n_epochs
        batch_size = int(config.batch_size)
        batches_per_epoch = int(len(cat_train) / batch_size)
        LOG.info('Batch Size = %i', batch_size)

        # Early stop stuff
        best_loss = 1E100
        best_epoch = -1

        # Grid search stuff
        wandb.watch(model, loss_fn, log="all")

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
                loss = loss_fn(cat_pred, cat_batch, cat_err_batch)
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = (epoch_loss / batches_per_epoch)
            model.eval()
            test_pred = model(torch.from_numpy(photo_test))
            test_loss = loss_fn(test_pred, torch.from_numpy(cat_test), torch.from_numpy(cat_err_test))
            LOG.info('Epoch %i/%i finished with avg training loss = %.3f', epoch + 1, n_epochs, avg_train_loss)

            # Always store best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_epoch = epoch
                wandb.log({'epoch': epoch + 1, 'loss': test_loss})
                checkpoint(model, best_model_path)

            # Early stopping
            elif epoch - best_epoch >= 50:
                LOG.info('Loss has not decreased in 50 epochs, early stopping. Best test loss is %.3f', best_loss)
                break

        # Load best model
        resume(model, best_model_path)
        LOG.info('!!!Finished Training!!!')
    tmp_dir.cleanup()


if __name__ == '__main__':
    train()

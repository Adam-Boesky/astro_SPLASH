import torch
import numpy as np

from neural_net import resume, get_model
from sklearn.ensemble import RandomForestClassifier

torch.set_default_dtype(torch.float64)  # set the pytorch default to a float64


class SnPipeline:
    def __init__(self, pipeline_version: str = 'full_band') -> None:

        # Check that we aren't given anything wrong
        if pipeline_version not in ['full_band', 'best_band', 'full_band_weighted', 'full_band_powlaw4_weighted', 'full_band_powlaw6_weighted', 'V2_full_band_powlaw6_weighted', 'full_band_n_weighted']:
            raise ValueError(f'{pipeline_version} is not an option for your pipeline version, it must either be \'full_band\' or \'best_band\' or \'full_band_weighted\'')

        # Which version of the classifier do we want to use
        self.version = pipeline_version

        # Load the hyperparameters of the given pipeline version
        if self.version == 'full_band':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_model.pkl'}
        elif self.version == 'best_band':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':9, 'nodes_per_layer': [6, 7, 7, 8], 'num_linear_output_layers': 2, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/V2_best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 14, 'num_outputs': 3, 'nodes_per_layer': [12, 10, 8, 6, 4], 'num_linear_output_layers': 3, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/V2_host_prop_best_model.pkl'}
        elif self.version == 'full_band_weighted':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/weighted_host_prop_best_model.pkl'}
        elif self.version == 'full_band_powlaw4_weighted':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/powlaw4_weighted_host_prop_best_model.pkl'}
        elif self.version == 'full_band_powlaw6_weighted':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/powlaw6_weighted_host_prop_best_model.pkl'}
        elif self.version == 'V2_full_band_powlaw6_weighted':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [16, 15, 12, 9, 6, 4], 'num_linear_output_layers': 2, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/V2_powlaw6_weighted_host_prop_best_model.pkl'}
        elif self.version == 'full_band_n_weighted':
            domain_transfer_hyperparams = {'num_inputs':5, 'num_outputs':13, 'nodes_per_layer': [7, 9, 11, 13], 'num_linear_output_layers': 1, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/best_sed_model.pkl'}
            host_prop_hyperparams = {'num_inputs': 18, 'num_outputs': 3, 'nodes_per_layer': [18, 15, 12, 9, 6, 4], 'num_linear_output_layers': 3, 'weights_fname': '/Users/adamboesky/Research/ay98/Weird_Galaxies/powlaw_n6_weighted_host_prop_best_model.pkl'}

        # Import domain transfer and host prop predicting NNs
        # 1. Domain transfer network
        self.domain_transfer_net = get_model(num_inputs=domain_transfer_hyperparams['num_inputs'], num_outputs=domain_transfer_hyperparams['num_outputs'], nodes_per_layer=domain_transfer_hyperparams['nodes_per_layer'], num_linear_output_layers=domain_transfer_hyperparams['num_linear_output_layers'])
        resume(self.domain_transfer_net, domain_transfer_hyperparams['weights_fname'])
        self.domain_transfer_net.eval()
        # 2. Host property network
        self.property_predicting_net = get_model(num_inputs=host_prop_hyperparams['num_inputs'], num_outputs=host_prop_hyperparams['num_outputs'], nodes_per_layer=host_prop_hyperparams['nodes_per_layer'], num_linear_output_layers=host_prop_hyperparams['num_linear_output_layers'])
        resume(self.property_predicting_net, host_prop_hyperparams['weights_fname'])
        self.property_predicting_net.eval()
        # 3. Classifier
        # self.random_forest = random_forest # TODO: FOR LATER :)

        # Variable that we can store host props in if we want to predict properties and then classify
        self.host_props = None

    def predict_host_properties(self, X_grizy: np.ndarray) -> np.ndarray:

        # Check that correct data format
        if X_grizy.shape[1] != 5:
            raise ValueError(f'Input grizy dimensions are {X_grizy.shape} when they should be (n, 5).')

        # Conduct domain transfer
        other_bands = self.domain_transfer_net(torch.from_numpy(X_grizy)).detach().numpy()
        X_full_band = np.hstack((X_grizy, other_bands))  # note that this is still scaled to be input into the host prop NN

        # Predict host properties
        self.host_props = self.property_predicting_net(torch.from_numpy(X_full_band)).detach().numpy()

        return self.host_props

    # def predict(self, X):
    #     # Transform data with domain transfer network
    #     X_transformed = self.domain_transfer_net.transform(X)

    #     # Get features for random forest
    #     features_for_rf = self.property_predicting_net.predict(X_transformed)

    #     # Predict with random forest
    #     return self.random_forest.predict(features_for_rf)

import numpy as np

from torch import nn, load
from typing import List, Optional
from astropy.cosmology import Planck18 as cosmo


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


def resume(model, filepath):
    """Resume pytorch model."""
    model.load_state_dict(load(filepath))


def get_intrinsic_mags(mags: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Get the intrinsic magnitudes of an object"""
    d_pc = cosmo.luminosity_distance(z).to('pc').value.reshape(-1, 1)
    return mags - 5 * np.log10(d_pc / 10)


def get_mag_at_z(intrinsic_mags: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Get the apparent magnitudes at a given redshift"""
    d_pc = cosmo.luminosity_distance(z).to('pc').value.reshape(-1, 1)
    return intrinsic_mags + 5 * np.log10(d_pc / 10)


def ab_mag_to_flux(AB_mag: np.ndarray, magerr: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert AB magnitude to flux in units of mJy"""
    flux = 10**((AB_mag - 8.9) / -2.5) * 1000
    if magerr is not None:
        fluxerr = (magerr * (np.log(10) * flux)) / 2.5
        return flux, fluxerr
    return flux


def flux_to_ab_mag(flux: np.ndarray, fluxerr: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert flux in units of mJy to AB magnitude"""
    mag = -2.5 * np.log10(flux / 1000) + 8.9
    if fluxerr is not None:
        magerr = (2.5 * fluxerr) / (np.log(10) * flux)
        return mag, magerr
    return mag

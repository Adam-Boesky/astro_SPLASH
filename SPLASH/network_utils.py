from torch import nn, load
from typing import List

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

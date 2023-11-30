"""Pytho script used to write the yaml config file for sweeps."""
import yaml


def write_config():
    """Function that writes a config yaml file."""
    config = {
        'method': 'grid',
        'metric': {'goal': 'minimize', 'name': 'loss'},
        'parameters': {
            'batch_size': {
                'values': [32, 64, 128, 256, 512, 1024, 2048, 4096]
            },
            'n_epochs': {'value': 1000},
            'nodes_per_layer': {'values': [[18, 13,  8,  4], [18, 14, 11,  7,  4], [18, 15, 12,  9,  6,  4]]},
            'num_linear_output_layers': {'values': [2,3]},
            'learning_rate': {'values': np.linspace(0.01, 1.0, num=5).tolist()}
        }
    }

    with open('/n/home04/aboesky/berger/Weird_Galaxies/sweep_config.yaml', 'w') as f:
        yaml.dump(config, f)


if __name__ == 'main':
    write_config()

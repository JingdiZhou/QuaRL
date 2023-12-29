# Define sweep config
sweep_configuration = {
    'method': 'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters':
        {
            'learning_rate': {'values': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]},
            'rho': {'values': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]},
        }
}

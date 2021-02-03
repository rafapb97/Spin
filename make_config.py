from snntoolbox.utils.utils import import_configparser
import os 

configparser = import_configparser()
config = configparser.ConfigParser()
path = os.path.abspath((os.path.dirname(os.path.realpath(__file__))))
config['paths'] = {
    'path_wd': path,             # Path to model.
    'dataset_path': path,        # Path to dataset.
    'filename_ann': "CNN1",     # Name of input model.
    'filename_snn': "sCNN1"
}

config['tools'] = {
    'evaluate_ann': True,           # Test ANN on dataset before conversion.
    # Normalize weights for full dynamic range.
    'normalize': False,
    'scale_weights_exp': False,
    'simulate': False
}

config['simulation'] = {
    # Chooses execution backend of SNN toolbox.
    'simulator': 'spiNNaker',
    'duration': 50,                 # Number of time steps to run each sample.
    'num_to_test': 1,               # How many test samples to run.
    'batch_size': 1,                # Batch size for simulation.
    # SpiNNaker seems to require 0.1 for comparable results.
    'dt': 0.1

    
}

config['input'] = {
    'poisson_input': True,           # Images are encodes as spike trains.
    'input_rate': 1000
}

config['cell'] = {
    'v_thresh' : 0.01,
    'tau_refrac' : 0.1,
    'v_reset' : 0,
    'v_rest' : 0,
    'cm' : 1,
    'tau_m' : 1000,
    'tau_syn_E' : 0.01,
    'tau_syn_I' : 0.01}

"""
config['output'] = {
    'plot_vars': {                  # Various plots (slows down simulation).
        'spiketrains',              # Leave section empty to turn off plots.
        'spikerates',
        'activations',
        'correlation',
        'v_mem',
        'error_t'}
}
"""
# Store config file.
config_filepath = 'config'
with open(config_filepath, 'w') as configfile:
    config.write(configfile)
    
    


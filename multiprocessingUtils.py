## Imports
# General
import os
import numpy as np
from shared_memory_dict import SharedMemoryDict
# Local
import train_MappingJM as trainMap


def worker_init(shared_dict, size):
    """
    Function to initialize workers with shared memory. Allows to avoid multiple copies
    of the whole datasets in RAM.

    Args:
      - shared_dict: string; Name of the shared dictionnary
      - size       : int   ; Expected size of the pickled file in bytes
    """
    ## Initialize global variable
    global _global_shared_dict

    ## Attach global variable to shared memory
    _global_shared_dict = SharedMemoryDict(name = shared_dict, size = size)

    print(f"[Worker Init] Attached to shared memory '{shared_dict}' (PID: {os.getpid()})")

def run_training(input_params):
    """
    Function running the training of the model for each given set of parameters.
    """
    ## Get values from dict
    model    = input_params.get('model')
    ablation = input_params.get('ablation')
    fold     = input_params.get('fold')
    nb_folds = input_params.get('nb_folds')
    data     = _global_shared_dict.get(ablation).get(fold)

    ## Build params dictionnary for each model
    params = {'nb_folds': nb_folds}

    ## Get trained models (k models for each type of fitting)
    fitted_models = trainMap.train_MappingJM(data, model, params = params)
    input_params.update({'fitted_models': fitted_models})

    return input_params
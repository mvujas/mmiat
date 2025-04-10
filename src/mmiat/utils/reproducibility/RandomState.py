import random
import contextlib

import torch
import numpy as np

@contextlib.contextmanager
def temp_seed(seed: int):
    """
    Creates a context with seeds set to given value. Returns to the
    previous seed afterwards. 

    Note: Based on torch implementation there might be issues with CUDA
        causing troubles with the correctness of this function. Function
        torch.rand() work fine from testing as their results are generated
        on CPU regardless if CUDA is used for other things.
        
    """
    random_state    = random.getstate()
    np_old_state    = np.random.get_state()
    torch_old_state = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_old_state)
        torch.random.set_rng_state(torch_old_state)


class RandomState:
    """
    Creates a state that affects random number generation on 
    torch and numpy and whose context can be activated at will

    """
    def __init__(self, seed: int):
        with temp_seed(seed):
            self.__refresh_states()

    def __refresh_states(self):
        self.__random_state = random.getstate()
        self.__np_state     = np.random.get_state()
        self.__torch_state  = torch.random.get_rng_state()

    def __set_states(self):
        random.setstate(self.__random_state)
        np.random.set_state(self.__np_state)
        torch.random.set_rng_state(self.__torch_state)

    @contextlib.contextmanager
    def activate(self):
        """
        Activates this state in the given context for torch and
        numpy. The previous state is restored when the context
        is finished

        """
        random_state    = random.getstate()
        np_old_state    = np.random.get_state()
        torch_old_state = torch.random.get_rng_state()
        self.__set_states()
        try:
            yield
        finally:
            self.__refresh_states()
            random.setstate(random_state)
            np.random.set_state(np_old_state)
            torch.random.set_rng_state(torch_old_state)
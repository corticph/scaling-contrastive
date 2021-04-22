import numpy as np
from scipy.io import wavfile
from tensorflow.python.client import device_lib

def get_device():
    """
    Returns of the first available device.
    
    Raises an error if there are no available GPUs.
    """
    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    assert len(gpus) > 0, "No GPU available."
    device_name = gpus[0]
    return device_name

def load_file(path):
    """
    Loads a .wav file, adds batch dim and depth and prepare the sequence length tensor.
    
    Args:
        path (str): Path to a .wav file.
    """
    _, x = wavfile.read("test.wav")
    x = x.reshape([1, -1, 1]) # batch_size x time x depth
    x_seq_len = np.array([x.shape[1]], dtype=np.int32)
    return x, x_seq_len
import numpy

epsilon = numpy.finfo(numpy.float32).eps

def normalize(v):
    l = numpy.linalg.norm(v, axis=-1)
    return v / numpy.expand_dims(l, len(l.shape))

class Data:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


# Define the suffixes for each size
byte_suffixes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB', 'ZiB', 'YiB']

def human_readable_bytes(num_bytes):
    # Initialize the index for the suffixes
    i = 0
    
    # Convert bytes to a human-readable format
    while num_bytes >= 1024 and i < len(byte_suffixes)-1:
        num_bytes /= 1024.0
        i += 1
    
    # Format the number with the appropriate suffix
    return (f'{num_bytes:.2f}' if i > 0 else f'{num_bytes}') + byte_suffixes[i]

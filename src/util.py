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

class ByteCounterStream:
    def __init__(self, stream):
        self._stream = stream
        self.bytes_read = 0

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._stream)
        self.bytes_read += len(line.encode(self._stream.encoding))
        return line

def mat4_rotation_x(theta):
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    return numpy.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ],
        dtype=numpy.float64
    )

def mat4_rotation_y(theta):
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    return numpy.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ],
        dtype=numpy.float64
    )

def mat4_rotation_z(theta):
    c = numpy.cos(theta)
    s = numpy.sin(theta)
    return numpy.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        dtype=numpy.float64
    )

import argparse, sys, pandas, numpy, numpy.ma
from mesh import Mesh
from index import MeshIndex
from PIL import Image
from logger import StderrLogger
from util import human_readable_bytes

job_dtype = numpy.dtype([
    ('pixel', '<2u4'),
    ('node', '<u4'),
    ('flags', '<u4'),
    ('ray_origin', '<3f4'),
    ('ray_direction', '<3f4'),
    ('ray_range', '<2f4'),
    ('normal', '<3f4'),
])

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

def main():
    numpy.set_printoptions(suppress=True)
    logger = StderrLogger()
    input_stream = ByteCounterStream(sys.stdin)

    logger.log('[stdin]: read/parse wavefront .obj file...')
    mesh = Mesh.from_wavefront_stream(input_stream)
    logger.log('[stdin]: done(bytes_read=%s, triangles=%s)' % (human_readable_bytes(input_stream.bytes_read), len(mesh.triangles[1:])))

    box_index = MeshIndex(mesh, logger=StderrLogger())

    database_count = max(box_index.database.keys())
    output_size = 32 + box_index.nodes.nbytes + sum(a.nbytes for a in box_index.database.values())
    logger.log('[stdout]: writing(bytes=%s)' % human_readable_bytes(output_size))
    byteorder = 'little'

    sys.stdout.buffer.write(b'\x7fd3i') # File signature (0x7F ensures file is always recognized as binary)
    sys.stdout.buffer.write((0).to_bytes(2, byteorder)) # Version major = 0 (experimental)
    sys.stdout.buffer.write((1).to_bytes(2, byteorder)) # Version minor = 1 (sequential)
    sys.stdout.buffer.write((len(box_index.nodes)).to_bytes(4, byteorder)) # Number of nodes
    sys.stdout.buffer.write((database_count).to_bytes(4, byteorder)) # Number of databases
    for idx in range(1, 1 + database_count):
        db = box_index.database[idx]
        sys.stdout.buffer.write(len(db).to_bytes(4, byteorder))
        sys.stdout.buffer.write(db.nbytes.to_bytes(4, byteorder))
    sys.stdout.buffer.write(box_index.nodes.tobytes())
    for idx in range(1, 1 + database_count):
        db = box_index.database[idx]
        sys.stdout.buffer.write(db.tobytes())

if __name__ == '__main__':
    sys.exit(main())

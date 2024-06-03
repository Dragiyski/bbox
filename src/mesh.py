import numpy
import pandas

vertex_type = numpy.dtype([('position', '3f4'), ('normal', '3f4'), ('texcoord', '2f4')])

class Mesh:
    def __init__(self, *, position_data = [[0, 0, 0]], normal_data = [[0, 0, 0]], texcoord_data = [[0, 0]], face_data = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]):
        position_data = numpy.array(position_data, dtype=numpy.float32)
        normal_data = numpy.array(normal_data, dtype=numpy.float32)
        texcoord_data = numpy.array(texcoord_data, dtype=numpy.float32)
        face_data = numpy.array(face_data, dtype=numpy.uint32)

        self.position = pandas.DataFrame(dict([(dim, position_data[:, idx]) for idx, dim in enumerate('xyz')]))
        self.normal = pandas.DataFrame(dict([(dim, normal_data[:, idx]) for idx, dim in enumerate('xyz')]))
        self.texcoord = pandas.DataFrame(dict([(dim, normal_data[:, idx]) for idx, dim in enumerate('uv')]))
        self.vertices, vertex_index = numpy.unique(face_data.reshape(-1, 3), return_inverse=True, axis=0)
        self.vertices = pandas.DataFrame(dict([(dim, self.vertices[:, idx]) for idx, dim in enumerate(['position', 'texcoord', 'normal'])]))
        self.face = pandas.DataFrame(dict([(idx, vertex_index.reshape(face_data.shape[:2])[:, idx].astype(numpy.uint32)) for idx in range(3)]))

        self.triangles = numpy.core.records.fromarrays([
            position_data[face_data[:, :, 0]],
            normal_data[face_data[:, :, 2]],
            texcoord_data[face_data[:, :, 1]],
        ], dtype=vertex_type)

    @classmethod
    def from_wavefront_stream(cls, stream):
        position_list = [[0, 0, 0]]
        normal_list = [[0, 0, 0]]
        texcoord_list = [[0, 0]]
        face_list = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        for line in stream:
            line = line.strip()
            if len(line) <= 0 or line.startswith('#'):
                continue
            first_hash = line.find('#')
            if first_hash >= 0:
                line = line[0:first_hash].strip()
            if len(line) <= 0:
                continue
            word_list = line.split()
            if word_list[0] == 'v':
                if len(word_list) > 4:
                    raise RuntimeError('Wavefront Object: expected "v" having 3 coordinates')
                position_list.append([float(x) for x in word_list[1:]])
            elif word_list[0] == 'vn':
                if len(word_list) > 4:
                    raise RuntimeError('Wavefront Object: expected "vn" having 3 coordinates')
                normal_list.append([float(x) for x in word_list[1:]])
            elif word_list[0] == 'vt':
                if len(word_list) > 3:
                    raise RuntimeError('Wavefront Object: expected "vt" having 2 coordinates')
                texcoord_list.append([float(x) for x in word_list[1:]])
            elif word_list[0] == 'f':
                if len(word_list) > 4:
                    raise RuntimeError('Wavefront Object: expected "f" have 3 coordinates, the file must be triangulaized')
                face = []
                for word in word_list[1:]:
                    index_list = [int(x) if len(x) > 0 else 0 for x in word.split('/')]
                    while len(index_list) < 3:
                        index_list.append(0)
                    face.append(index_list)
                face_list.append(face)
        return cls(position_data=position_list, normal_data=normal_list, texcoord_data=texcoord_list, face_data=face_list)
    
    @classmethod
    def from_wavefront_file(cls, file):
        with open(file, 'r') as stream:
            return cls.from_wavefront_stream(stream)
import argparse, sys, pandas, numpy, numpy.ma
from mesh import Mesh
from PIL import Image

def normalize(v):
    l = numpy.linalg.norm(v, axis=-1)
    return v / numpy.expand_dims(l, len(l.shape))

def main():
    numpy.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Experiment for 3D index suitable for raytracing"
    )
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()
    mesh = Mesh.from_wavefront_stream(args.file)
    index = MeshIndex(mesh)

    # Camera Setup: 1 per frame, outside GPU
    pixel_width = 800
    pixel_height = 600
    aspect_ratio = pixel_width / pixel_height
    camera_position = numpy.array([0.5, -2.0, 0.5], dtype=numpy.float32)
    camera_direction = normalize(-camera_position)
    world_up = numpy.array([0.0, 0.0, 1.0], dtype=numpy.float32)
    view_right = normalize(numpy.cross(camera_direction, world_up))
    view_up = normalize(numpy.cross(view_right, camera_direction))
    field_of_view = numpy.radians(60)
    diagonal = numpy.tan(field_of_view * 0.5)
    screen_height = diagonal / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
    screen_width = aspect_ratio * screen_height
    screen_right = screen_width * view_right
    screen_up = screen_height * view_up
    screen_center = camera_position + camera_direction
    ray_origin = camera_position

    # Ray generation: 1 per pixel, in GPU
    view_pixels = numpy.moveaxis(numpy.mgrid[0:pixel_height, 0:pixel_width], 0, -1).astype(numpy.float32)
    view_pixels[:, :, 1] = (view_pixels[:, :, 1] + 0.5) / pixel_width
    view_pixels[:, :, 0] = (view_pixels[:, :, 0] + 0.5) / pixel_height
    view_pixels[:, :, 0] = 1.0 - view_pixels[:, :, 0]
    view_pixels = view_pixels * 2.0 - 1.0
    world_pixels = screen_center + view_pixels[:, :, 0, None] * screen_up + view_pixels[:, :, 1, None] * screen_right
    ray_direction = normalize(world_pixels - camera_position)

    bounding_box = numpy.vstack([index.nodes[index.root_node]['position_min'], index.nodes[index.root_node]['position_max']])
    ray_distance = (bounding_box - ray_origin)[None, None] / ray_direction[:, :, None]
    intersection_points = ray_distance[:, :, :, :, None] * ray_direction[:, :, None, None] + ray_origin
    bounding_box_size = bounding_box[1] - bounding_box[0]
    box_ray_quad = (intersection_points - bounding_box[0]) / bounding_box_size
    for d in range(3):
        box_ray_quad[:, :, 0, d, d] = 0.0
        box_ray_quad[:, :, 1, d, d] = 1.0
    box_intersection = numpy.logical_and(numpy.all(box_ray_quad >= 0.0, axis=-1), numpy.all(box_ray_quad <= 1.0, axis=-1)).reshape(box_ray_quad.shape[0], box_ray_quad.shape[1], -1)
    has_intersection = numpy.any(box_intersection, axis=-1)
    
    box_ray_isect_distance = numpy.ma.array(ray_distance[has_intersection].reshape(-1, 6), mask=numpy.logical_not(box_intersection[has_intersection]))
    box_ray_isect_side = box_ray_isect_distance.argmin(axis=-1)
    box_ray_isect_side = numpy.stack([box_ray_isect_side % 3, box_ray_isect_side // 3], axis=-1)
    box_color = numpy.zeros(shape=(box_ray_isect_side.shape[0], 3), dtype=numpy.float32)
    box_color[(numpy.arange(box_color.shape[0]), box_ray_isect_side[:, 0])] = box_ray_isect_side[:, 1] * 2.0 - 1.0
    box_color = box_color * 0.5 + 0.5
    assert box_ray_isect_side.shape[0] == box_ray_isect_distance.shape[0], 'box_ray_isect_side.shape[0] == box_ray_isect_distance.shape[0]'
    pass
    
    color = numpy.zeros((pixel_height, pixel_width, 3), dtype=numpy.float32)
    color[has_intersection] = box_color
    color = (numpy.clip(color, 0.0, 1.0) * 255.0).astype(numpy.uint8)
    image = Image.fromarray(color)
    image.show()
    pass

    # sys.stdout.buffer.write(b'\x7fdgi') # Signature
    # sys.stdout.buffer.write(int(1).to_bytes(4, 'little')) # Version
    # sys.stdout.buffer.write(len(mesh.triangles).to_bytes(4, 'little')) # Number of triangles
    # sys.stdout.buffer.write(len(index.nodes).to_bytes(4, 'little')) # Number of nodes
    # sys.stdout.buffer.write(index.depth.to_bytes(4, 'little')) # Depth of tree
    # sys.stdout.buffer.write(int(1).to_bytes(4, 'little')) # Pointer to root node
    # sys.stdout.buffer.write(int(0).to_bytes(8, 'little')) # Padding
    # sys.stdout.buffer.write(mesh.triangles.astype(export_dtype).view('8f4').tobytes())
    pass

class MeshIndex2:
    def __init__(self, mesh: Mesh):
        triangle_data = pandas.DataFrame(dict([('%d.%s' % (idx, dim), mesh.position.loc[mesh.vertices.loc[mesh.face.loc[1:, idx], 'position'], dim].reset_index(drop=True)) for idx in range(3) for dim in 'xyz']))
        triangle_data.index = mesh.face.index[1:]
        group_data = pandas.DataFrame(dict([('%s.%s' % (lim, dim), triangle_data[['%d.%s' % (idx, dim) for idx in range(3)]].T.agg(lim)) for lim in ['min', 'max'] for dim in 'xyz']))
        groups = pandas.DataFrame(dict(group_data.agg(dict([('%s.%s' % (lim, dim), lim) for lim in ['min', 'max'] for dim in 'xyz'])).items()), index=[numpy.uint32(1)])
        group_data['group'] = numpy.uint32(1)
        pass


class MeshIndex:
    _lim = ['min', 'max']

    @staticmethod
    def test(*args, **kwargs):
        return None

    def __init__(self, mesh: Mesh):
        triangle_data = pandas.DataFrame(dict([('%d.%s' % (idx, dim), mesh.position.loc[mesh.vertices.loc[mesh.face.loc[1:, idx], 'position'], dim].reset_index(drop=True)) for idx in range(3) for dim in 'xyz']))
        triangle_data.index = mesh.face.index[1:]
        data = pandas.DataFrame(dict([('%s.%s' % (lim, dim), triangle_data[['%d.%s' % (idx, dim) for idx in range(3)]].T.agg(lim)) for lim in ['min', 'max'] for dim in 'xyz']))
        data.index = data.index.astype('UInt32')
        self.mesh = mesh
        self.data = pandas.DataFrame({'group': pandas.Series(1, index=data.index, dtype='UInt32')})
        self.data[data.columns] = data
        self.nodes = pandas.DataFrame(dict([('%s.%s' % (dir, dim), data['%s.%s' % (dir, dim)].agg(dir)) for dir in ['min', 'max'] for dim in 'xyz' ] + [('count', pandas.Series([len(data)], dtype='UInt32', index=[1]))]), index=pandas.Index([1], name='group'))
        self.nodes['parent_node'] = self.nodes['next_sibling'] = self.nodes['first_child'] = pandas.Series(0, dtype='UInt32', index=self.nodes.index)
        self.depth = 0
        for _ in range(int(numpy.ceil(numpy.log2(len(self.data))))):
            self.depth += 1
            node_list = self.nodes.loc[self.data['group']].reset_index(drop=True)
            node_list.index = self.data.index
            affinity_list = pandas.DataFrame(dict([('group', self.data['group'])] + [('subgroup.%s' % dim, numpy.uint32(0)) for dim in 'xyz'] + [('affinity.%s' % dim, (self.data['min.%s' % dim] + self.data['max.%s' % dim]) - (node_list['min.%s' % dim] + node_list['max.%s' % dim])) for dim in 'xyz']))
            
            group_count = affinity_list.groupby('group')['group'].agg('count')
            large_group_count = group_count[group_count > 6]
            if len(large_group_count) <= 0:
                break
            large_group_mask = affinity_list['group'].isin(large_group_count.index)
            minmax_data = self.data[large_group_mask]
            affinity_list = affinity_list[large_group_mask]
            affinity_sort = pandas.DataFrame(dict([(dim, affinity_list.sort_values(['group', 'affinity.%s' % dim]).index) for dim in 'xyz']), index=affinity_list.index)
            affinity_sort_groupby = dict([(dim, minmax_data.loc[affinity_sort[dim]].groupby('group')) for dim in 'xyz'])
            affinity_index = dict([('min.%s' % dim, affinity_sort_groupby[dim].apply(lambda g: g[:len(g) >> 1], include_groups=False).index.get_level_values(1).values) for dim in 'xyz'] + [('max.%s' % dim, affinity_sort_groupby[dim].apply(lambda g: g[len(g) >> 1:], include_groups=False).index.get_level_values(1).values) for dim in 'xyz'])
            for dim in 'xyz':
                affinity_list.loc[affinity_index['min.%s' % dim], 'subgroup.%s' % dim] = 1
                affinity_list.loc[affinity_index['max.%s' % dim], 'subgroup.%s' % dim] = 2
            group_overlap = pandas.DataFrame(dict([('%s.%s' % (self._lim[d], dim), minmax_data.loc[affinity_index['%s.%s' % (self._lim[1-d], dim)]].groupby('group')['%s.%s' % (self._lim[d], dim)].agg(self._lim[d])) for d in range(2) for dim in 'xyz']))
            group_overlap_data = group_overlap.loc[minmax_data['group']].reset_index(drop=True)
            group_overlap_data.index = minmax_data.index
            min_group_overlap = dict([(dim, (minmax_data.loc[affinity_index['min.%s' % dim], 'max.%s' % dim] - group_overlap_data.loc[affinity_index['min.%s' % dim], 'min.%s' % dim])) for dim in 'xyz'])
            max_group_overlap = dict([(dim, (group_overlap_data.loc[affinity_index['max.%s' % dim], 'max.%s' % dim] - minmax_data.loc[affinity_index['max.%s' % dim], 'min.%s' % dim])) for dim in 'xyz'])
            group_overlap = pandas.DataFrame(dict([(dim, pandas.concat([min_group_overlap[dim], max_group_overlap[dim]], axis=0)) for dim in 'xyz']))
            group_overlap['group'] = affinity_list['group']
            group_overlap_count = pandas.DataFrame(dict([(dim, group_overlap[group_overlap[dim] >= 0].groupby('group')[dim].agg('count').astype('UInt32')) for dim in 'xyz']))
            group_overlap_count.fillna(0, inplace=True)
            group_overlap_count.columns = range(3)
            split_dim = group_overlap_count.idxmin(axis=1)
            overlap_column = split_dim[group_overlap['group']].reset_index(drop=True)
            overlap_column.index = group_overlap.index
            group_overlap['overlap'] = pandas.Series(group_overlap[['x', 'y', 'z']].values[numpy.arange(len(group_overlap)), split_dim.loc[group_overlap['group']].values], index=group_overlap.index)
            affinity_list['subgroup'] = pandas.Series(affinity_list[['subgroup.%s' % dim for dim in 'xyz']].values[numpy.arange(len(affinity_list)), split_dim.loc[affinity_list['group']].values], index=affinity_list.index)
            group_overlap_sort = group_overlap.sort_values(['group', 'overlap'], ascending=[True, False])
            mix_group = group_overlap_sort.groupby('group')['overlap'].apply(lambda g: g[g >= 0].head(len(g) // 3))
            affinity_list.loc[mix_group.index.get_level_values(1), 'subgroup'] = 3
            separation_data = pandas.DataFrame(minmax_data.loc[affinity_list.index])
            separation_data['subgroup'] = affinity_list['subgroup']
            separation_data_groupby = separation_data.groupby(['group', 'subgroup'])
            new_nodes = separation_data_groupby.agg(dict([('%s.%s' % (lim, dim), lim) for lim in ['min', 'max'] for dim in 'xyz'] + [('subgroup', 'count')]))
            new_nodes.rename(columns={'subgroup': 'count'}, inplace=True)
            new_nodes['count'] = new_nodes['count'].astype('uint32')
            new_nodes['node_index'] = (numpy.arange(len(new_nodes)) + self.nodes.index.max() + 1).astype('uint32')
            new_nodes['parent_node'] = new_nodes.index.get_level_values(0)
            new_nodes_by_group = new_nodes.groupby(level=0)
            new_nodes.loc[new_nodes_by_group.head(-1).index, 'next_sibling'] = pandas.Series(new_nodes_by_group.tail(-1)['node_index'].values, dtype='UInt32').values
            new_nodes.fillna({'next_sibling': 0}, inplace=True)
            new_nodes['first_child'] = numpy.uint32(0)
            new_nodes['first_child'] = new_nodes['first_child'].astype('UInt32')
            self.nodes.loc[new_nodes_by_group['parent_node'].head(1).values, 'first_child'] = new_nodes_by_group['node_index'].head(1).values
            append_nodes = pandas.DataFrame(new_nodes[self.nodes.columns].reset_index(drop=True))
            append_nodes.index = new_nodes['node_index']
            self.nodes = pandas.concat([self.nodes, append_nodes], axis=0)
            separation_index = numpy.vstack([separation_data.index.values, separation_data['group'].values, separation_data['subgroup'].values]).T
            separation_groups, separation_num_index = numpy.unique(separation_index[:, 1:], return_inverse=True, axis=0)
            separation_node_index = new_nodes.loc[list(zip(separation_groups[:, 0], separation_groups[:, 1]))]['node_index'].values
            self.data.loc[separation_data.index, 'group'] = separation_node_index[separation_num_index]
        self.root_node = 1
        self.nodes.loc[0] = {k: v.type(0) for k, v in self.nodes.dtypes.items()}
        self.nodes.sort_index(inplace=True)
        nodes_src = self.nodes.to_records(index=False)
        nodes_dtype = numpy.dtype([
            ('position_min', '<3f4'),
            ('position_max', '<3f4'),
            ('count', '<u4'),
            ('parent_node', '<u4'),
            ('next_sibling', '<u4'),
            ('first_child', '<u4')
        ])
        nodes_dst = numpy.recarray(nodes_src.shape, dtype=nodes_dtype)
        nodes_dst['position_min'] = numpy.stack((nodes_src['min.x'], nodes_src['min.y'], nodes_src['min.z']), axis=-1)
        nodes_dst['position_max'] = numpy.stack((nodes_src['max.x'], nodes_src['max.y'], nodes_src['max.z']), axis=-1)
        nodes_dst['count'] = nodes_src['count']
        nodes_dst['parent_node'] = nodes_src['parent_node']
        nodes_dst['next_sibling'] = nodes_src['next_sibling']
        nodes_dst['first_child'] = nodes_src['first_child']
        self.nodes = nodes_dst
        pass


if __name__ == '__main__':
    sys.exit(main())

import argparse, sys, pandas, numpy
from mesh import Mesh

def main():
    numpy.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Experiment for 3D index suitable for raytracing"
    )
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()
    mesh = Mesh.from_wavefront_stream(args.file)
    index = MeshIndex(mesh)

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
    lim = ['min', 'max']

    @staticmethod
    def test(*args, **kwargs):
        return None

    def __init__(self, mesh: Mesh):
        triangle_data = pandas.DataFrame(dict([('%d.%s' % (idx, dim), mesh.position.loc[mesh.vertices.loc[mesh.face.loc[1:, idx], 'position'], dim].reset_index(drop=True)) for idx in range(3) for dim in 'xyz']))
        triangle_data.index = mesh.face.index[1:]
        data = pandas.DataFrame(dict([('%s.%s' % (lim, dim), triangle_data[['%d.%s' % (idx, dim) for idx in range(3)]].T.agg(lim)) for lim in ['min', 'max'] for dim in 'xyz']))
        #mesh_min = mesh.triangles.position[1:].min(axis=1)
        #esh_max = mesh.triangles.position[1:].max(axis=1)
        # data = pandas.DataFrame({
        #     'min.x': mesh_min[:, 0],
        #     'min.y': mesh_min[:, 1],
        #     'min.z': mesh_min[:, 2],
        #     'max.x': mesh_max[:, 0],
        #     'max.y': mesh_max[:, 1],
        #     'max.z': mesh_max[:, 2],
        # })
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
            group_overlap = pandas.DataFrame(dict([('%s.%s' % (self.lim[d], dim), minmax_data.loc[affinity_index['%s.%s' % (self.lim[1-d], dim)]].groupby('group')['%s.%s' % (self.lim[d], dim)].agg(self.lim[d])) for d in range(2) for dim in 'xyz']))
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
        # self.nodes.loc[0] = dict([('%s.%s' % (dir, dim), numpy.float32(0.0)) for dir in ['min', 'max'] for dim in 'xyz'] + [('count', numpy.uint32(0)), ('parent_node', numpy.uint32(0)), ('next_sibling', numpy.uint32(0)), ('first_child', numpy.uint32(0))])
        # self.nodes.index = self.nodes.index.astype('UInt32')
        # self.export_bbox = self.nodes[['%s.%s' % (dir, dim) for dir in ['min', 'max'] for dim in 'xyz']].sort_index().to_records(index=False)
        # self.export_triangles = mesh.triangles
        # self.tree_data = pandas.DataFrame({'parent_node': self.data['group'].values, 'next_sibling': numpy.uint32(0), 'first_child': numpy.uint32(0), 'count': numpy.uint32(1)}, index=self.data.index + len(self.nodes))
        # self.tree_data['next_sibling'] = self.tree_data.groupby('parent_node')['next_sibling'].transform(lambda g: numpy.hstack([g.index[1:].values, [0]]).astype(numpy.uint32))
        # pass
        # How to organize this data?:
        # - We need to add to self.nodes for leaf nodes an access to triangles.
        # - We need somehow to separate nodes by type, with bounding boxes handled by one code, while triangles handled by another.
        # - We need general metadata - max tree depth
        # How the data would be used?
        # - Initially a ray is shot once per pixel, this is Width X Height number of jobs
        # - A job takes a node (initially all jobs pick the root node)
        # - A job can schedule 0 or more jobs to do after the current job:
        #   - Newly scheduled jobs must be able to be processed out-of-order and in parallel.
        #   - Memory must be organized in a structure where:
        #   - Different kind of jobs would be added with different data size, but still can be accessed in parallel?
        # Big problem: How to organize the memory?
        # 1. It must be assumed that the number of types of jobs would be small (256 or 65536 types max)
        # 2. We can reserve one large amount of memory to hold the buffer (unfortunately, otherwise there is no way to reserve buffer whose size is given by the GPU)
        # 3. Using rotational buffer as memory: we allocate large enough fixed size buffer (Let's say 1GiB). We can allocate to memory a space in that buffer.
        # 4. Jobs stored in the memory must contain both source data and space to write the result.
        # 5. Execution of jobs in memory will write the result in each space within the job, and use atomicAdd to count the number of jobs about to be scheduled, separated by type.
        # 6. A single shader invocation (1, 1, 1) takes the atomic values, and computes the amount of new memory required for the new jobs (i.e. its position in the memory), and compute the new job size.
        # 7. A shader invocation with old jobs form copy the old memory and the static data to the new job space.
        # 8. A single shader invocation (1, 1, 1) "releases" the old space to be reused.
        # 9. New jobs executes.

        # That is, we need a single buffer. Initial stage will have the following to form uniform data jobs:
        # A single root node (the same for all pixels);
        # A pixel/ray data;
        # Because the root node is the same for all pixels (thus the type is the same), and because we shoot 1 ray (for now) per pixel,
        # the initial memory space for jobs can be precomputed on the CPU. This can be loaded in the atomic memory.

        # A raytracing job must contain the following:
        # - A pointer to node
        # - ray_min_distance
        # - ray_max_distance
        # - screen position: x, y
        # - ray origin: vec3
        # - ray direction: vec3


if __name__ == '__main__':
    sys.exit(main())

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
    pass


class MeshIndex:
    def __init__(self, mesh: Mesh):
        mesh_min = mesh.triangles.position[1:].min(axis=1)
        mesh_max = mesh.triangles.position[1:].max(axis=1)
        data = pandas.DataFrame({
            'min.x': mesh_min[:, 0],
            'min.y': mesh_min[:, 1],
            'min.z': mesh_min[:, 2],
            'max.x': mesh_max[:, 0],
            'max.y': mesh_max[:, 1],
            'max.z': mesh_max[:, 2],
        })
        self.data = pandas.DataFrame({'group': pandas.Series(1, index=data.index)})
        self.data[data.columns] = data
        self.nodes = pandas.DataFrame(dict([('%s.%s' % (dir, dim), data['%s.%s' % (dir, dim)].agg(dir)) for dir in ['min', 'max'] for dim in 'xyz' ] + [('count', pandas.Series([len(data)], dtype='Int64', index=[1]))]), index=pandas.Index([1], name='group'))
        self.nodes['parent_node'] = self.nodes['next_sibling'] = self.nodes['first_child'] = pandas.Series(0, dtype='Int64', index=self.nodes.index)
        
        for _ in range(int(numpy.ceil(numpy.log2(len(self.data))))):
            node_list = self.nodes.loc[self.data['group']].reset_index(drop=True)
            affinity_list = pandas.DataFrame(dict([('group', self.data['group'])] + [('affinity.%s' % dim, (self.data['min.%s' % dim] + self.data['max.%s' % dim]) - (node_list['min.%s' % dim] + node_list['max.%s' % dim])) for dim in 'xyz']))
            
            group_count = affinity_list.groupby('group')['group'].agg('count')
            large_group_count = group_count[group_count > 6]
            if len(large_group_count) <= 0:
                break
            large_group_mask = affinity_list['group'].isin(large_group_count.index)
            large_group_data = self.data[large_group_mask]
            affinity_list = affinity_list[large_group_mask]
            for dim in 'xyz':
                affinity_sort = affinity_list.sort_values(['group', 'affinity.%s' % dim])
                affinity_list['group.%s' % dim] = pandas.Series(dtype='Int64')
                affinity_sort_groupby = affinity_sort.groupby('group')
                min_groups = affinity_sort_groupby.apply(lambda g: g[:len(g) >> 1], include_groups=False).index.get_level_values(1)
                max_groups = affinity_sort_groupby.apply(lambda g: g[len(g) >> 1:], include_groups=False).index.get_level_values(1)
                affinity_list.loc[min_groups, 'group.%s' % dim] = 1
                affinity_list.loc[max_groups, 'group.%s' % dim] = 2
                pass
            group_overlap_limit = large_group_count // 3
            large_group_overlap = pandas.DataFrame(dict([('%s.%s' % (dir, dim), large_group_data.loc[affinity_list[affinity_list['group.%s' % dim] == g].index].groupby('group')['%s.%s' % (dir, dim)].agg(dir)) for g, dir in {2: 'min', 1: 'max'}.items() for dim in 'xyz']))
            large_group_overlap_items = large_group_overlap.loc[large_group_data['group']].reset_index(drop=True)
            large_group_overlap_items.index = large_group_data.index
            for dim in 'xyz':
                min_group_index = affinity_list.index[affinity_list['group.%s' % dim] == 1]
                max_group_index = affinity_list.index[affinity_list['group.%s' % dim] == 2]
                affinity_list['overlap.%s' % dim] = pandas.concat([
                    (large_group_data['max.%s' % dim] - large_group_overlap_items['min.%s' % dim]).loc[min_group_index],
                    (large_group_overlap_items['max.%s' % dim] - large_group_data['min.%s' % dim]).loc[max_group_index]
                ], axis=0)
                overlap_index = affinity_list[affinity_list['overlap.%s' % dim] >= 0].sort_values(['group', 'overlap.%s' % dim], ascending=[True, False]).groupby('group').apply(lambda g: g.head(group_overlap_limit[g.name]), include_groups=False).index.get_level_values(1)
                affinity_list.loc[overlap_index, 'group.%s' % dim] = 3
            large_group_overlap_count = pandas.DataFrame(dict([(dim, affinity_list[affinity_list['overlap.%s' % dim] >= 0].groupby('group')['group'].agg('count')) for dim in 'xyz']))
            large_group_split_dim = large_group_overlap_count.idxmin(axis=1)
            next_group = pandas.concat([g['group.%s' % large_group_split_dim[n]] for n, g in affinity_list.groupby('group')], axis=0)
            next_group = pandas.DataFrame({'group': affinity_list['group'].loc[next_group.index].values, 'subgroup': next_group.values}, index=next_group.index)
            for (g, sg), i in next_group.groupby(['group', 'subgroup']):
                node = self.data.loc[i.index].agg(dict([('%s.%s' % (dir, dim), dir) for dir in ['min', 'max'] for dim in 'xyz']))
                gi = self.nodes.index.max() + 1
                node = pandas.DataFrame(dict(zip(node.axes[0].values, node.values)), index=[gi])
                node['parent_node'] = pandas.Series(g, dtype=self.nodes.dtypes['parent_node'], index=node.index)
                node['next_sibling'] = pandas.Series(0, dtype=self.nodes.dtypes['next_sibling'], index=node.index)
                node['first_child'] = pandas.Series(0, dtype=self.nodes.dtypes['next_sibling'], index=node.index)
                node['count'] = pandas.Series(len(i), dtype=self.nodes.dtypes['next_sibling'], index=node.index)
                prev_node = self.nodes[numpy.logical_and(self.nodes['parent_node'] == g, self.nodes['next_sibling'] == 0)]
                if len(prev_node) > 0:
                    assert len(prev_node) == 1, 'len(prev_node) == 1'
                    self.nodes.loc[prev_node.index[0], 'next_sibling'] = node.index[0]
                else:
                    assert self.nodes.loc[g, 'first_child'] == 0, "self.nodes.loc[g, 'first_child'] == 0"
                    self.nodes.loc[g, 'first_child'] = node.index[0]
                self.nodes = pandas.concat([self.nodes, node])
                self.data.loc[i.index, 'group'] = gi
        print(self.nodes)
        print(self.data)
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

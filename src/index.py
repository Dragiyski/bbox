import numpy, pandas
from mesh import Mesh

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

        self.data['index'] = self.data.index
        self.data.index += len(self.nodes) - 1
        data_first_child = self.data.groupby('group').head(1)
        self.nodes.loc[data_first_child['group'], 'first_child'] = data_first_child.index
        self.data.rename(columns={'group': 'parent_node'}, inplace=True)
        self.data['first_child'] = numpy.uint32(0)
        self.data['next_sibling'] = numpy.uint32(0)
        self.data.sort_values('parent_node', inplace=True)
        data_node_groups = self.data.groupby('parent_node')
        self.data.loc[data_node_groups.head(-1).index, 'next_sibling'] = data_node_groups.tail(-1).index
        self.data['count'] = numpy.uint32(1)
        self.data.sort_index(inplace=True)
        pass

        nodes_src = self.nodes.to_records(index=False)
        nodes_dtype = numpy.dtype([
            ('type', '<u4'),
            ('ref', '<u4'),
            ('leaf_count', '<u4'),
            ('parent_node', '<u4'),
            ('next_sibling', '<u4'),
            ('first_child', '<u4'),
            ('reserved1', '<u4'),
            ('reserved2', '<u4')
        ])
        self.database = [None]
        self.database.append(numpy.stack([
            numpy.vstack([self.nodes['min.%s' % dim].values for dim in 'xyz']).T,
            numpy.vstack([self.nodes['max.%s' % dim].values for dim in 'xyz']).T
        ], axis=1))
        self.database.append(mesh.triangles)
        bbox_nodes = numpy.recarray(tuple([len(self.nodes)]), dtype=nodes_dtype)
        bbox_nodes['type'] = 1
        bbox_nodes['ref'] = self.nodes.index.values
        bbox_nodes['leaf_count'] = self.nodes['count']
        bbox_nodes['parent_node'] = self.nodes['parent_node']
        bbox_nodes['next_sibling'] = self.nodes['next_sibling']
        bbox_nodes['first_child'] = self.nodes['first_child']
        bbox_nodes['reserved1'] = bbox_nodes['reserved2'] = 0
        trig_nodes = numpy.recarray(tuple([len(self.data)]), dtype=nodes_dtype)
        trig_nodes['type'] = 2
        trig_nodes['ref'] = self.data['index']
        trig_nodes['leaf_count'] = self.data['count']
        trig_nodes['parent_node'] = self.data['parent_node']
        trig_nodes['next_sibling'] = self.data['next_sibling']
        trig_nodes['first_child'] = self.data['first_child']
        trig_nodes['reserved1'] = trig_nodes['reserved2'] = 0
        self.nodes = numpy.hstack([bbox_nodes, trig_nodes])
        del self.data
        pass
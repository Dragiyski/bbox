import numpy, pandas, sys
from mesh import Mesh
from logger import NullLogger

lims = ['min', 'max']

def init_row(data, columns, dtypes):
    assert len(dtypes) == len(columns), 'len(dtypes) == len(columns)'
    return dict([(column, pandas.Series(data[column] if column in data else None, name=column, dtype=dtypes[i])) for i, column in enumerate(columns)])

class MeshIndex:
    node_dtype = numpy.dtype({
        'names': ['type', 'ref', 'size', 'parent_node', 'first_child', 'next_sibling'],
        'formats': ['<u4', '<u4', '<u4', '<u4', '<u4', '<u4'],
        'offsets': numpy.arange(6) * 4,
        'itemsize': 8 * 4
    })

    def __init__(self, mesh: Mesh, *, logger=NullLogger(), bounding_box_type = 1, triangle_type = 2):
        logger.log('[indexing]: initializing(triangles=%d)' % len(mesh.triangles[1:]))
        self.database = {
            triangle_type: mesh.triangles,
            bounding_box_type: numpy.ndarray((mesh.triangles.shape[0] * 2 + 1, 2, 3), dtype=numpy.float32)
        }
        self.database[bounding_box_type][0, 0] = [-numpy.inf] * 3
        self.database[bounding_box_type][0, 1] = [numpy.inf] * 3
        triangle_minmax = numpy.stack([mesh.triangles[1:]['position'].min(axis=1), mesh.triangles[1:]['position'].max(axis=1)], axis=-2)
        self.database[bounding_box_type][1:1+triangle_minmax.shape[0]] = triangle_minmax
        bounding_box_count = 1 + triangle_minmax.shape[0]
        self.nodes = numpy.recarray((triangle_minmax.shape[0] * 2 + 1,), dtype=self.node_dtype)
        self.nodes[0] = tuple([0] * len(self.node_dtype.names))
        self.nodes['parent_node'] = 0
        self.nodes['first_child'] = 0
        self.nodes['next_sibling'] = 0
        self.nodes['type'][1:1+triangle_minmax.shape[0]] = triangle_type
        self.nodes['ref'][1:1+triangle_minmax.shape[0]] = numpy.arange(triangle_minmax.shape[0]) + 1
        self.nodes['size'][1:1+triangle_minmax.shape[0]] = 1
        root_node = node_count = 1 + triangle_minmax.shape[0] 
        self.nodes[root_node] = tuple([bounding_box_type, bounding_box_count, triangle_minmax.shape[0], 0, 0, 0])
        node_count += 1
        self.database[bounding_box_type][bounding_box_count] = numpy.vstack([triangle_minmax[:, 0].min(axis=0), triangle_minmax[:, 1].max(axis=0)])
        bounding_box_count += 1

        grouping = pandas.DataFrame({
            'group': pandas.Series(root_node, dtype='UInt32', index=numpy.arange(triangle_minmax.shape[0]) + 1),
            'affinity.x': pandas.Series(dtype='Float32'),
            'affinity.y': pandas.Series(dtype='Float32'),
            'affinity.z': pandas.Series(dtype='Float32'),
            'subgroup': pandas.Series(dtype='UInt32'),
            'subgroup.x': pandas.Series(dtype='UInt32'),
            'subgroup.y': pandas.Series(dtype='UInt32'),
            'subgroup.z': pandas.Series(dtype='UInt32'),
            'overlap': pandas.Series(dtype='Float32'),
            'overlap.x': pandas.Series(dtype='Float32'),
            'overlap.y': pandas.Series(dtype='Float32'),
            'overlap.z': pandas.Series(dtype='Float32'),
            'ref': pandas.Series(numpy.arange(triangle_minmax.shape[0]) + 1, dtype='UInt32', index=numpy.arange(triangle_minmax.shape[0]) + 1)
        } | dict([('%s.%s' % (lim, dim), pandas.Series(triangle_minmax[:, ilim, idim], dtype='Float32', index=numpy.arange(triangle_minmax.shape[0]) + 1)) for ilim, lim in enumerate(lims) for idim, dim in enumerate('xyz')]), index=numpy.arange(triangle_minmax.shape[0]) + 1)
        grouping.index.name = 'index'
        self.depth = 0
        for level in range(int(numpy.ceil(numpy.log2(len(triangle_minmax))))):
            self.depth += 1
            group_count = grouping.groupby('group', sort=False)['group'].agg('count')
            large_mask = group_count > 6
            grouping_large = grouping[large_mask.loc[grouping['group']].values]
            logger.log('[indexing]: splitting(depth=%d, triangles=%d, nodes=%d)' % (self.depth, len(grouping_large), node_count))
            if len(grouping_large) <= 0:
                break
            affinity = (self.database[bounding_box_type][grouping_large.index, 0] + self.database[bounding_box_type][grouping_large.index, 1]) - (self.database[bounding_box_type][grouping_large['group'], 0] + self.database[bounding_box_type][grouping_large['group'], 1])
            for idx, dim in enumerate('xyz'):
                grouping_large.loc[grouping_large.index, 'affinity.%s' % dim] = affinity[:, idx]
                affinity_sort = grouping_large.sort_values(['group', 'affinity.%s' % dim])
                affinity_sort_group = affinity_sort.groupby('group', sort=False)
                min_group = affinity_sort_group.apply(lambda g: g[:len(g) >> 1])
                max_group = affinity_sort_group.apply(lambda g: g[len(g) >> 1:])
                grouping_large.loc[min_group.index.get_level_values(1), 'subgroup.%s' % dim] = 1
                grouping_large.loc[max_group.index.get_level_values(1), 'subgroup.%s' % dim] = 2
                grouping_large.loc[min_group.index.get_level_values(1), 'overlap.%s' % dim] = self.database[bounding_box_type][min_group.index.get_level_values(1)][:, 1, idx]
                grouping_large.loc[max_group.index.get_level_values(1), 'overlap.%s' % dim] = self.database[bounding_box_type][max_group.index.get_level_values(1)][:, 0, idx]
                affinity_subgroup = grouping_large.groupby(['group', 'subgroup.%s' % dim], sort=False)
                overlap = affinity_subgroup['overlap.%s' % dim].agg(['min', 'max'])
                grouping_large.loc[min_group.index.get_level_values(1), 'overlap.%s' % dim] = self.database[bounding_box_type][min_group.index.get_level_values(1)][:, 1, idx] - overlap.loc[grouping_large.loc[min_group.index.get_level_values(1)]['group']].xs(2, level=1)['min'].values
                grouping_large.loc[max_group.index.get_level_values(1), 'overlap.%s' % dim] = overlap.loc[grouping_large.loc[max_group.index.get_level_values(1)]['group']].xs(1, level=1)['max'].values - self.database[bounding_box_type][max_group.index.get_level_values(1)][:, 0, idx]
                pass
            is_overlap = pandas.concat([grouping_large['group'], grouping_large[['overlap.%s' % dim for dim in 'xyz']] >= 0.0], axis=1)
            is_overlap_group = is_overlap.groupby('group')
            overlap_count = is_overlap_group.sum()
            overlap_count.columns = range(3)
            split_dim = overlap_count.idxmin(axis=1)
            split_column = split_dim.loc[grouping_large['group']]
            grouping_large.loc[grouping_large.index, 'overlap'] = grouping_large[['overlap.%s' % dim for dim in 'xyz']].values[numpy.arange(len(grouping_large.index)), split_column.values]
            grouping_large.loc[grouping_large.index, 'subgroup'] = grouping_large[['subgroup.%s' % dim for dim in 'xyz']].values[numpy.arange(len(grouping_large.index)), split_column.values]
            overlap_mask = grouping_large['overlap'] >= 0.0
            overlap_rows = grouping_large[overlap_mask].groupby('group', sort=False).apply(lambda g: g.head(group_count.loc[g.name] // 3))
            grouping_large.loc[overlap_rows.index.get_level_values(1), 'subgroup'] = 3
            grouping_sort = grouping_large.sort_values(['group', 'subgroup', 'index'])
            grouping_split = grouping_sort.groupby(['group', 'subgroup'], sort=False)
            grouping_split_box = grouping_split.agg(dict([('%s.%s' % (lim, dim), lim) for lim in lims for dim in 'xyz']) | {'subgroup': 'count'})
            grouping_split_box.rename(columns={'subgroup': 'size'}, inplace=True)
            grouping_split_box['box_ref'] = bounding_box_count + numpy.arange(len(grouping_split_box))
            grouping_split_box['node_ref'] = node_count + numpy.arange(len(grouping_split_box))
            self.database[bounding_box_type][grouping_split_box['box_ref']] = grouping_split_box[['%s.%s' % (lim, dim) for lim in lims for dim in 'xyz']].astype(numpy.float32).values.reshape(-1, 2, 3)
            pass
            self.nodes['type'][grouping_split_box['node_ref']] = bounding_box_type
            self.nodes['ref'][grouping_split_box['node_ref']] = grouping_split_box['box_ref']
            self.nodes['size'][grouping_split_box['node_ref']] = grouping_split_box['size'].astype(numpy.uint32)
            self.nodes['parent_node'][grouping_split_box['node_ref']] = grouping_split_box.index.get_level_values(0).astype(numpy.uint32)
            self.nodes['first_child'][grouping_split_box['node_ref']] = 0
            self.nodes['next_sibling'][grouping_split_box['node_ref']] = 0
            grouping_first_child = grouping_split_box.xs(1, level=1)['node_ref']
            assert numpy.all(self.nodes[grouping_first_child.index.astype(numpy.uint32)]['first_child'] == 0)
            self.nodes['first_child'][grouping_first_child.index.astype(numpy.uint32)] = grouping_first_child
            grouping_split_box_group = grouping_split_box.groupby('group', sort=False)
            self.nodes['next_sibling'][grouping_split_box_group.head(-1)['node_ref']] = grouping_split_box_group.tail(-1)['node_ref']
            bounding_box_count += len(grouping_split_box)
            node_count += len(grouping_split_box)
            grouping.loc[grouping_large.index, 'group'] = grouping_split_box.loc[pandas.MultiIndex.from_frame(grouping_large[['group', 'subgroup']])]['node_ref'].values
            pass
        self.database[bounding_box_type] = self.database[bounding_box_type][:bounding_box_count]
        self.nodes = self.nodes[:node_count]
        logger.log('[indexing]: done(depth=%d, triangles=%d, nodes=%d)' % (self.depth, len(mesh.triangles[1:]), len(self.nodes)))

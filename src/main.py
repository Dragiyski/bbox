import argparse, sys, pandas, numpy, numpy.ma
from mesh import Mesh
from index import MeshIndex
from PIL import Image

job_dtype = numpy.dtype([
    ('screen_position', '<2u4'),
    ('node_index', '<u4'),
    ('flags', '<u4'),
    ('ray_origin', '<3f4'),
    ('ray_direction', '<3f4'),
    ('ray_range', '<2f4'),
    ('normal', '<3f4'),
])

epsilon = numpy.finfo(numpy.float32).eps

def normalize(v):
    l = numpy.linalg.norm(v, axis=-1)
    return v / numpy.expand_dims(l, len(l.shape))

class Raytracer:
    def __init__(self, index: MeshIndex, width=800, height=600, field_of_view=60, camera_position=[0.0, -1.0, 0.0], world_up=[0.0, 0.0, 1.0]):
        self.camera_position = numpy.array(camera_position, dtype=numpy.float32)
        if len(self.camera_position.shape) != 1 or self.camera_position.shape[0] != 3:
            raise ValueError('Option "camera_position" must be 3D vector')
        self.camera_direction = normalize(-self.camera_position)
        self.width = width
        self.height = height
        self.index = index

        world_up = numpy.array(world_up, dtype=numpy.float32)
        if len(world_up.shape) != 1 or world_up.shape[0] != 3:
            raise ValueError('Option "world_up" must be 3D vector')
        world_up = normalize(world_up)
        view_right = normalize(numpy.cross(self.camera_direction, world_up))
        view_up = normalize(numpy.cross(view_right, self.camera_direction))

        diagonal = numpy.tan(numpy.radians(field_of_view) * 0.5)
        aspect_ratio = self.width / self.height
        screen_height = diagonal / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
        self.world_screen_width = aspect_ratio * screen_height
        self.world_screen_right = self.world_screen_width * view_right
        self.world_screen_up = screen_height * view_up
        self.world_screen_center = self.camera_position + self.camera_direction
        self.jobs_by_type = {k: numpy.empty((0,), dtype=job_dtype) for k in numpy.unique(index.nodes['type'])}

    def intersect_bounding_box(self):
        jobs = self.jobs_by_type[1]
        if jobs.shape[0] <= 0:
            return
        bounding_box = self.index.database[1][jobs['node_index']]
        ray_distance = (bounding_box - jobs['ray_origin'][:, None]) / jobs['ray_direction'][:, None]
        box_ray_points = ray_distance[:, :, :, None] * jobs['ray_direction'][:, None, None] + jobs['ray_origin'][:, None, None]
        box_size = bounding_box[:, 1] - bounding_box[:, 0]
        box_ray_quad = (box_ray_points - bounding_box[:, 0][:, None, None]) / box_size[:, None, None]
        for d in range(3):
            box_ray_quad[:, 0, d, d] = 0.0
            box_ray_quad[:, 1, d, d] = 1.0
        box_intersection = numpy.logical_and(numpy.all(box_ray_quad >= 0.0, axis=-1), numpy.all(box_ray_quad <= 1.0, axis=-1)).reshape(box_ray_quad.shape[0], -1)
        has_intersection = numpy.any(box_intersection, axis=-1)
        box_ray_distance = numpy.ma.array(ray_distance[has_intersection].reshape(-1, 6), mask=numpy.logical_not(box_intersection[has_intersection]))
        box_ray_min = box_ray_distance.min(axis=-1)
        assert not numpy.any(box_ray_min.mask), 'not numpy.any(box_ray_min.mask)'
        box_ray_max = box_ray_distance.max(axis=-1)
        assert not numpy.any(box_ray_max.mask), 'not numpy.any(box_ray_max.mask)'
        # ray_range will contain the previous ray range (initialized to 0-inf for primary rays).
        # box_ray_min and box_ray_max contain the new ray range such that:
        # Bn --- Bx     Rn --- Rx
        # Bn --- Rn +++ Bx --- Rx
        # Bn --- Rn +++ Rx --- Bx
        # Rn --- Bn +++ Rx --- Bx
        # Rn --- Bn +++ Bx --- Rx
        # Rn --- Rx     Bn --- Bx
        # where n is min, x is max, B is before range, R is the raytraced new range
        # that is B is jobs['ray_range'] and R is box_ray_min/box_ray_max
        # The two cases where previous and new range do not intersect must be excluded.
        
        # IMPORTANT: It does not seem to get any cases like  Bn --- Bx     Rn --- Rx, and Rn --- Rx     Bn --- Bx
        # The reason is because all bounding boxes and triangles are contained within the parent bounding box
        # While those cases would only happen if something is intersected outside of the bounding box
        # Therefore the range_mask optimization is redundant (it does not optimize anything).
        
        jobs['normal'] = 0.0
        normal_index = box_ray_distance.argmin(axis=-1)
        jobs['normal'][numpy.nonzero(has_intersection)[0], normal_index % 3] = numpy.float32(normal_index // 3) * 2 - 1
        # range_mask = numpy.logical_or(jobs['ray_range'][has_intersection][:, 1] < box_ray_min.data, jobs['ray_range'][has_intersection][:, 0] > box_ray_max.data)
        # print('range_mask.count = %d' % numpy.count_nonzero(range_mask))
        # has_intersection[has_intersection][range_mask] = False
        # jobs['ray_range'][has_intersection] = numpy.stack([
        #     numpy.maximum(box_ray_min.data[numpy.logical_not(range_mask)], jobs['ray_range'][has_intersection][:, 0]),
        #     numpy.minimum(box_ray_max.data[numpy.logical_not(range_mask)], jobs['ray_range'][has_intersection][:, 1]),
        # ], axis=-1)
        jobs['ray_range'][has_intersection] = numpy.stack([
            numpy.maximum(box_ray_min.data, jobs['ray_range'][has_intersection][:, 0]) - epsilon,
            numpy.minimum(box_ray_max.data, jobs['ray_range'][has_intersection][:, 1]) + epsilon,
        ], axis=-1)
        jobs['flags'] = has_intersection
        pass

    def insersect_triangles(self):
        jobs = self.jobs_by_type[2]
        if jobs.shape[0] <= 0:
            return
        # This intersection does not work, the base shape is there but some triangles are missing:
        # 1. It is not because the order of triangle vertices;
        # 2. It does not seem to be related to the bounding box epsilon mismatch.
        # 3. This work quite differntly from the stack version, in the stack version the dot(cross(Edge, Point - Opposite Vertex), Normal)
        # seems to provide with something that sums up to 1.0, but here barycentric items do not sum to 1.0.
        triangles = self.index.database[2][self.index.nodes[jobs['node_index']]['ref']]['position']
        triangle_normal = normalize(numpy.cross(triangles[:, 2] - triangles[:, 0], triangles[:, 1] - triangles[:, 0]))
        # triangle_normal_degenerate_mask = numpy.abs(numpy.repeat(numpy.float32(1.0), 3) / triangle_normal) < epsilon
        plane_distance = (numpy.einsum('ab,ab->a', triangle_normal, triangles[:, 0]) - numpy.einsum('ab,ab->a', triangle_normal, jobs['ray_origin'])) / numpy.einsum('ab,ab->a', triangle_normal, jobs['ray_direction'])
        ray_range_mask = numpy.logical_and(plane_distance >= jobs['ray_range'][:, 0], plane_distance <= jobs['ray_range'][:, 1])
        plane_point = jobs['ray_origin'] + plane_distance[:, None] * jobs['ray_direction']
        
        barycentric = numpy.stack([
            numpy.einsum('ab,ab->a', numpy.cross(triangles[:, 1] - triangles[:, 2], plane_point - triangles[:, 2]), triangle_normal),
            numpy.einsum('ab,ab->a', numpy.cross(triangles[:, 2] - triangles[:, 0], plane_point - triangles[:, 0]), triangle_normal),
            numpy.einsum('ab,ab->a', numpy.cross(triangles[:, 0] - triangles[:, 1], plane_point - triangles[:, 1]), triangle_normal)
        ], axis=-1)
        test_barycentry = numpy.stack([
            numpy.einsum('ab,ab->a', numpy.cross(normalize(triangles[:, 1] - triangles[:, 2]), normalize(plane_point - triangles[:, 2])), triangle_normal),
            numpy.einsum('ab,ab->a', numpy.cross(normalize(triangles[:, 2] - triangles[:, 0]), normalize(plane_point - triangles[:, 0])), triangle_normal),
            numpy.einsum('ab,ab->a', numpy.cross(normalize(triangles[:, 0] - triangles[:, 1]), normalize(plane_point - triangles[:, 1])), triangle_normal),
        ], axis=-1)
        barycentric = numpy.sign(barycentric.sum(axis=-1))[:, None] * barycentric
        has_intersection = numpy.all(barycentric >= 0.0, axis=-1)
        jobs['flags'] = has_intersection
        jobs['ray_range'] = plane_distance[:, None]
        jobs['normal'] = ((numpy.einsum('ab,ab->a', triangle_normal, -jobs['ray_direction']) >= 0.0).astype(numpy.float32) * 2.0 - 1.0)[:, None] * triangle_normal

        return

        normal_factor = numpy.sign(numpy.einsum('ij,ij->i', -jobs['ray_direction'], triangle_normal))
        normal_factor[normal_factor == 0.0] = 1.0
        triangle_view_normal = normal_factor[:, None] * triangle_normal
        # This is dot(triangle_normal, triangle first point), however numpy.dot does not perform per element dot product
        # like any other numpy operator - it performs matrix multiplication (which is dot product between first arg columns and second arg rows)
        # We cannot also do numpy.dot(A, B.T), because this will compute A.shape[0] * B.shape[1] values instead of A.shape[0] == B.shape[0]
        # Copilot suggested einsum, which apparently is efficient, but it contains a "magic" first argument to perform dot product.
        plane_factor = numpy.einsum('ij,ij->i', triangle_normal, triangles[:, 0])
        ray_distance = (plane_factor - numpy.einsum('ij,ij->i', triangle_normal, jobs['ray_origin'])) / numpy.einsum('ij,ij->i', triangle_normal, jobs['ray_direction'])
        # ray_distance_mask = ray_distance >= numpy.logical_and(jobs['ray_range'][:, 0], ray_distance <= jobs['ray_range'][:, 1])
        ray_distance_mask = ray_distance >= 0.0
        plane_jobs = jobs[ray_distance_mask]
        ray_distance = ray_distance[ray_distance_mask]
        triangles = triangles[ray_distance_mask]
        triangle_view_normal = triangle_view_normal[ray_distance_mask]
        triangle_normal = triangle_normal[ray_distance_mask]
        ray_point = plane_jobs['ray_origin'] + ray_distance[:, None] * plane_jobs['ray_direction']
        # ray_point = jobs['ray_origin'] + ray_distance[:, None] * jobs['ray_direction']
        ray_traingle_points = numpy.stack([ray_point - triangles[:, dim] for dim in range(3)], axis=1)
        triangle_cross = numpy.cross(ray_traingle_points[:, [0, 1, 2]], ray_traingle_points[:, [1, 2, 0]])
        triangle_mixed = numpy.einsum('ijk,ik->ij', triangle_cross, triangle_normal)
        triangle_inside = triangle_mixed >= 0.0
        has_intersection = numpy.all(triangle_inside, axis=-1)

        # jobs['flags'] = has_intersection
        # jobs['ray_range'] = ray_distance[:, None]
        # jobs['normal'] = triangle_normal
        
        jobs['flags'][numpy.logical_not(ray_distance_mask)] = 0
        jobs['flags'][ray_distance_mask] = has_intersection
        jobs['ray_range'][numpy.where(ray_distance_mask)[0][has_intersection]] = ray_distance[has_intersection, None]
        jobs['normal'][numpy.where(ray_distance_mask)[0][has_intersection]] = triangle_view_normal[has_intersection]
        pass

    def draw_jobs(self):
        jobs = numpy.hstack(list(self.jobs_by_type.values()))
        active_job_mask = (jobs['flags'] & 1).astype(bool)
        active_jobs = jobs[active_job_mask]
        pixel_distance = pandas.DataFrame({'y': active_jobs['screen_position'][:, 0], 'x': active_jobs['screen_position'][:, 1], 'd': active_jobs['ray_range'][:, 0]})
        pixel_depth_resolver = pixel_distance.groupby(['y', 'x'])['d'].idxmin()
        depth_mask = active_jobs['ray_range'][pixel_depth_resolver.values, 0] <= self.depth_buffer[pixel_depth_resolver.index.get_level_values(0).values, pixel_depth_resolver.index.get_level_values(1).values]
        pixel_jobs = active_jobs[pixel_depth_resolver.values][depth_mask]
        # self.color_buffer[active_jobs['screen_position'][:, 0], active_jobs['screen_position'][:, 1]] = active_jobs['normal'] * 0.5 + 0.5
        self.color_buffer[pixel_jobs['screen_position'][:, 0], pixel_jobs['screen_position'][:, 1]] = pixel_jobs['normal'] * 0.5 + 0.5
        # self.depth_buffer[active_jobs['screen_position'][:, 0], active_jobs['screen_position'][:, 1]] = active_jobs['ray_range'][:, 0]
        self.depth_buffer[pixel_jobs['screen_position'][:, 0], pixel_jobs['screen_position'][:, 1]] = pixel_jobs['ray_range'][:, 0]
        pass

    def schedule_children(self):
        next_jobs = []
        # jobs = numpy.hstack(list(self.jobs_by_type.values()))
        # numpy.argwhere(numpy.logical_and(jobs['screen_position'][:, 0] == 240, jobs['screen_position'][:, 1] == 440))
        # pixel 440, 240 must be present
        active_job_mask = (self.jobs_by_type[1]['flags'] & 1).astype(bool)
        active_jobs = self.jobs_by_type[1][active_job_mask]
        # while loop
        while True:
            i = len(next_jobs)
            s = 'next_sibling' if i > 0 else 'first_child'
            active_job_nodes = self.index.nodes[active_jobs['node_index']]
            node_mask = active_job_nodes[s] > 0
            if not numpy.any(node_mask):
                break
            next_jobs.append(active_jobs[node_mask])
            next_jobs[i]['node_index'] = self.index.nodes[next_jobs[i]['node_index']][s]
            active_jobs = next_jobs[i]
        # Tree nodes will have first_child = 0 when they contain only triangles.
        # Once the index is finished those should contain raytracing nodes (triangles for now)
        # So some of the boxes will be skipped for now
        # when no more jobs, return
        if len(next_jobs) > 0:
            jobs = numpy.hstack(next_jobs)
        else:
            jobs = numpy.empty((0,), dtype=job_dtype)
        self.jobs_by_type[1] = numpy.empty((0,), dtype=job_dtype)
        for t in self.jobs_by_type.keys():
            type_mask = self.index.nodes[jobs['node_index']]['type'] == t
            new_jobs_by_type = jobs[type_mask]
            if len(new_jobs_by_type) > 0:
                self.jobs_by_type[t] = numpy.hstack([self.jobs_by_type[t], new_jobs_by_type])
        print({k: len(v) for k, v in self.jobs_by_type.items()})
        pass

    def draw_bounding_boxes(self):
        active_job_mask = (self.jobs_by_type[1]['flags'] & 1).astype(bool)
        active_jobs = self.jobs_by_type[1][active_job_mask]
        box_ray_distance = active_jobs['ray_range'][:, 0]

        # This is only necessary for python, as pandas is more effective than a loop. In GPU, depth-buffer comparison happens in parallel.
        # Note: it might need atomics, but atomics are only integers, depth is float.
        pixel_distance = pandas.DataFrame({'y': active_jobs['screen_position'][:, 0], 'x': active_jobs['screen_position'][:, 1], 'd': box_ray_distance})
        pixel_depth_resolver = pixel_distance.groupby(['y', 'x'])['d'].idxmin()
        depth_mask = box_ray_distance[pixel_depth_resolver.values] <= self.depth_buffer[pixel_depth_resolver.index.get_level_values(0).values, pixel_depth_resolver.index.get_level_values(1).values]
        pixel_jobs = active_jobs[pixel_depth_resolver.values][depth_mask]
        self.color_buffer[pixel_jobs['screen_position'][:, 0], pixel_jobs['screen_position'][:, 1]] = pixel_jobs['normal'] * 0.5 + 0.5
        self.depth_buffer[pixel_jobs['screen_position'][:, 0], pixel_jobs['screen_position'][:, 1]] = box_ray_distance[pixel_depth_resolver.values][depth_mask]

    def display_color_buffer(self):
        color_buffer = (numpy.clip(self.color_buffer, 0.0, 1.0) * 255.0).astype(numpy.uint8)
        image = Image.fromarray(color_buffer)
        image.show()
    
    def run(self):
        self.color_buffer = numpy.zeros((self.height, self.width, 3), dtype=numpy.float32)
        self.depth_buffer = numpy.full((self.height, self.width), numpy.float32(numpy.inf))

        ray_origin = self.camera_position
        view_pixels = numpy.moveaxis(numpy.mgrid[0:self.height, 0:self.width], 0, -1).astype(numpy.float32)
        view_pixels[:, :, 1] = (view_pixels[:, :, 1] + 0.5) / self.width
        view_pixels[:, :, 0] = (view_pixels[:, :, 0] + 0.5) / self.height
        view_pixels[:, :, 0] = 1.0 - view_pixels[:, :, 0]
        view_pixels = view_pixels * 2.0 - 1.0
        world_pixels = self.world_screen_center + view_pixels[:, :, 0, None] * self.world_screen_up + view_pixels[:, :, 1, None] * self.world_screen_right
        ray_direction = normalize(world_pixels - ray_origin)

        self.jobs_by_type[1] = numpy.recarray(self.width * self.height, job_dtype)
        self.jobs_by_type[1]['screen_position'] = numpy.vstack([numpy.repeat(numpy.arange(self.height), self.width), numpy.tile(numpy.arange(self.width), self.height)]).T
        self.jobs_by_type[1]['node_index'] = 1
        self.jobs_by_type[1]['ray_origin'] = ray_origin
        self.jobs_by_type[1]['ray_direction'] = ray_direction.reshape(-1, 3)
        self.jobs_by_type[1]['ray_range'] = numpy.array([numpy.float32(0.0), numpy.float32(numpy.inf)])


        for _ in range(self.index.depth):
            self.intersect_bounding_box()
            self.schedule_children()
        # assert len(self.jobs_by_type[1]) == 0, 'len(self.jobs_by_type[1]) == 0'

        self.insersect_triangles()
        self.draw_jobs()
        self.display_color_buffer()
        pass

def main():
    numpy.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Experiment for 3D index suitable for raytracing"
    )
    parser.add_argument('file', type=argparse.FileType('r'))
    args = parser.parse_args()
    mesh = Mesh.from_wavefront_stream(args.file)
    index = MeshIndex(mesh)

    raytracer = Raytracer(index, camera_position=[0.5, -2.0, 0.5])
    raytracer.run()

if __name__ == '__main__':
    sys.exit(main())

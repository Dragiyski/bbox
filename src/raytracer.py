import sys, numpy, pandas
from index import MeshIndex
from mesh import Mesh
from util import Data, normalize, open_file_for_writing
from logger import NullLogger
from PIL import Image

job_dtype = numpy.dtype([
    ('pixel', '<2u4'),
    ('node', '<u4'),
    ('flags', '<u4'),
    ('ray_origin', '<3f4'),
    ('ray_direction', '<3f4'),
    ('ray_range', '<2f4'),
    ('normal', '<3f4'),
    ('ray_box', '<3f4', 2)
])

epsilon = numpy.finfo(numpy.float32).eps

class Raytracer:
    def __init__(self, index, logger=NullLogger()):
        self.index = index
        self.logger = logger
        self.jobs_by_type = { job_type: numpy.empty((0,), dtype=job_dtype) for job_type in numpy.unique(index.nodes['type']) }
        self.raytrace_per_type = { job_type: 0 for job_type in self.jobs_by_type.keys() }

    @staticmethod
    def _process_camera_options(aspect_ratio, *, field_of_view=60, camera_position=[0.0, -1.0, 0.0], world_up=[0.0, 0.0, 1.0], camera_direction=None, camera_focus_at=None, **kwargs):
        camera_position = numpy.array(camera_position, dtype=numpy.float32)
        if len(camera_position.shape) != 1 or camera_position.shape[0] != 3:
            raise ValueError('Option "camera_position" must be 3D vector')
        if camera_direction is None and camera_focus_at is None:
            camera_focus_at = numpy.zeros((3,), dtype=numpy.float32)
        if camera_focus_at is not None and camera_direction is not None:
            raise ValueError('Conflicting options "camera_direction" and "camera_focus_at"')
        if camera_focus_at is not None:
            camera_focus_at = numpy.array(camera_focus_at, dtype=numpy.float32)
            if len(camera_focus_at.shape) != 1 or camera_focus_at.shape[0] != 3:
                raise ValueError('Option "camera_focus_at" must be 3D vector')
            camera_direction = camera_focus_at - camera_position
        if camera_direction is not None:
            camera_direction = numpy.array(camera_direction, dtype=numpy.float32)
            if len(camera_direction.shape) != 1 or camera_direction.shape[0] != 3:
                raise ValueError('Option "camera_direction" must be 3D vector')
            camera_direction = normalize(camera_direction)
        world_up = numpy.array(world_up, dtype=numpy.float32)
        if len(world_up.shape) != 1 or world_up.shape[0] != 3:
            raise ValueError('Option "world_up" must be 3D vector')


        diagonal = numpy.tan(numpy.radians(field_of_view) * 0.5)
        screen_height = diagonal / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
        screen_width = aspect_ratio * screen_height

        world_up = normalize(world_up)
        view_right = normalize(numpy.cross(camera_direction, world_up))
        view_up = normalize(numpy.cross(view_right, camera_direction))

        world_screen_up = screen_height * view_up
        world_screen_right = screen_width * view_right
        world_screen_center = camera_position + camera_direction

        return Data(
            position=camera_position,
            direction=camera_direction,
            view_up=view_up,
            view_right=view_right,
            world_up=world_up,
            world_screen_up=world_screen_up,
            world_screen_right=world_screen_right,
            world_screen_diagonal=diagonal,
            world_screen_center=world_screen_center,
            field_of_view=field_of_view,
            aspect_ratio=aspect_ratio,
            world_screen_width=screen_width,
            world_screen_height=screen_height,
            camera_focus_at=camera_focus_at
        )

    def screen(self, width=800, height=600, **kwargs):
        self.camera = self._process_camera_options(width / height, **kwargs)
        self.color_buffer = numpy.zeros((height, width, 3), dtype=numpy.float32)
        self.depth_buffer = numpy.full((height, width), numpy.float32(numpy.inf))

        view_pixels = numpy.moveaxis(numpy.mgrid[0:height, 0:width], 0, -1).astype(numpy.float32)
        view_pixels[:, :, 1] = (view_pixels[:, :, 1] + 0.5) / width
        view_pixels[:, :, 0] = (view_pixels[:, :, 0] + 0.5) / height
        view_pixels[:, :, 0] = 1.0 - view_pixels[:, :, 0]
        view_pixels = view_pixels * 2.0 - 1.0
        world_pixels = self.camera.world_screen_center + view_pixels[:, :, 0, None] * self.camera.world_screen_up + view_pixels[:, :, 1, None] * self.camera.world_screen_right
        ray_direction = normalize(world_pixels - self.camera.position)

        view_transform = numpy.identity(4)
        view_transform[:3, 0] = self.camera.view_right
        view_transform[:3, 1] = self.camera.view_up
        view_transform[:3, 2] = self.camera.direction
        view_transform[:3, 3] = self.camera.position
        view_transform = numpy.linalg.inv(view_transform)

        # model_bounding_boxes = self.index.database[1][1:]
        # model_bounding_vertices = numpy.zeros((model_bounding_boxes.shape[0], 8, 3), model_bounding_boxes.dtype)
        # for i in range(8):
        #     ib = [int(bit) for bit in f'{i:03b}']
        #     model_bounding_vertices[:, i, :] = numpy.stack([model_bounding_boxes[:, ib[j], j] for j in range(3)], axis=-1)
        # view_bounding_vertices = numpy.einsum(
        #     'ij,pqj->pqi',
        #     view_transform,
        #     numpy.concatenate([model_bounding_vertices, numpy.ones(model_bounding_vertices.shape[:2], model_bounding_vertices.dtype)[:, :, None]], axis=-1)
        # )
        # view_bounding_boxes = numpy.stack([view_bounding_vertices.min(axis=1), view_bounding_vertices.max(axis=1)], axis=1)
        # root_view = view_bounding_boxes[self.index.nodes[self.index.root_node]['ref']]
        # root_view_match = numpy.logical_and.reduce([view_pixels[:, :, 0] >= root_view[0, 0], view_pixels[:, :, 0] <= root_view[1, 0], view_pixels[:, :, 1] >= root_view[0, 1], view_pixels[:, :, 1] <= root_view[1, 1]])
        # self.color_buffer[numpy.nonzero(root_view_match)] = [1.0, 1.0, 1.0]

        self.jobs_by_type[1] = numpy.recarray(width * height, job_dtype)
        self.jobs_by_type[1]['pixel'] = numpy.vstack([numpy.repeat(numpy.arange(height), width), numpy.tile(numpy.arange(width), height)]).T
        self.jobs_by_type[1]['node'] = self.index.root_node
        self.jobs_by_type[1]['ray_origin'] = self.camera.position
        self.jobs_by_type[1]['ray_direction'] = ray_direction.reshape(-1, 3)
        self.jobs_by_type[1]['ray_range'] = numpy.array([numpy.float32(0.0), numpy.float32(numpy.inf)])
        self.jobs_by_type[1]['ray_box'] = self.index.database[1][0]

        # self.color_buffer[self.jobs_by_type[1]['pixel'][:, 0], self.jobs_by_type[1]['pixel'][:, 1]] = self.jobs_by_type[1]['ray_direction'] * 0.5 + 0.5

        self.raytrace(**kwargs)
    
    def pixel(self, x, y, width=800, height=600, **kwargs):
        if x <= 0 or x >= width or y <= 0 or y >= height:
            raise ValueError('Invald pixel coordinates, expected (x, y) within (width, height)')
        
        camera = self._process_camera_options(width / height, **kwargs)
        self.color_buffer = numpy.zeros((1, 1, 3), dtype=numpy.float32)
        self.depth_buffer = numpy.full((1, 1), numpy.float32(numpy.inf))

        view_pixels = numpy.array([y, x], dtype=numpy.float32)[None]
        view_pixels[:, 1] = (view_pixels[:, 1] + 0.5) / width
        view_pixels[:, 0] = (view_pixels[:, 0] + 0.5) / height
        view_pixels[:, 0] = 1.0 - view_pixels[:, 0]
        view_pixels = view_pixels * 2.0 - 1.0
        world_pixels = camera.world_screen_center + view_pixels[:, 0, None] * camera.world_screen_up + view_pixels[:, 1, None] * camera.world_screen_right
        ray_direction = normalize(world_pixels - camera.position)

        self.jobs_by_type[1] = numpy.recarray((1,), job_dtype)
        self.jobs_by_type[1]['pixel'][0] = [0, 0]
        self.jobs_by_type[1]['node'][0] = self.index.root_node
        self.jobs_by_type[1]['ray_origin'][0] = camera.position
        self.jobs_by_type[1]['ray_direction'] = ray_direction.reshape(-1, 3)
        self.jobs_by_type[1]['ray_range'] = numpy.array([numpy.float32(0.0), numpy.float32(numpy.inf)])

        ray_origin = camera.position
        ray_direction = ray_direction[0]

        T = self.index.database[2][1:]['position']
        T_edges = T[:, [1, 2, 0]] - T
        T_normal = normalize(numpy.cross(T_edges[:, 0], T_edges[:, 1]))
        T_plane_factor = numpy.einsum('ab,ab->a', T_normal, T[:, 0])
        T_distance = numpy.einsum('ab,ab->a', T_normal, (T_plane_factor[:, None] - ray_origin)) / numpy.dot(T_normal, ray_direction)
        T_point = ray_origin + T_distance[:, None] * ray_direction
        T_inner = T_point[:, None] - T
        T_trig_intersection = numpy.stack([numpy.einsum('ab,ab->a', T_normal, numpy.cross(T_edges[:, i], T_inner[:, i])) for i in range(3)], axis=-1)
        T_has_intersection = numpy.all(T_trig_intersection >= 0.0, axis=-1)
        v0 = T[:, 1] - T[:, 0]
        v1 = T[:, 2] - T[:, 0]
        v2 = T_point - T[:, 0]
        dot00 = numpy.einsum('ab,ab->a', v0, v0)
        dot01 = numpy.einsum('ab,ab->a', v0, v1)
        dot02 = numpy.einsum('ab,ab->a', v0, v2)
        dot11 = numpy.einsum('ab,ab->a', v1, v1)
        dot12 = numpy.einsum('ab,ab->a', v1, v2)
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        w = 1 - u - v
        barycentric = numpy.stack([u, v, w], axis=-1)

        # For some reason only triangle 34 intersects
        # Node path: 288, 246, 93, 31, 10, 3, 1, 0

        pass

        self.raytrace(**kwargs)

    
    def raytrace(self, max_depth=None, draw_jobs=True, **kwargs):
        if max_depth is None:
            max_depth = self.index.depth
        for depth in range(max_depth):
            self.depth = depth
            self.logger.log('[raytracer] depth=%d: jobs: %r' % (self.depth, {k: v.shape[0] for k, v in self.jobs_by_type.items()}))
            self.raytrace_per_type[1] += self.jobs_by_type[1].shape[0]
            self.logger.log('[raytracer] depth=%d: Raytracing %d bounding boxes...' % (self.depth, self.jobs_by_type[1].shape[0]))
            self.intersect_bounding_box()
            self.schedule_children()
        self.depth = depth

        self.logger.log('[raytracer] depth=%d: jobs: %r' % (self.depth, {k: v.shape[0] for k, v in self.jobs_by_type.items()}))
        self.logger.log('[raytracer] depth=%d: Raytracing %d triangles...' % (self.depth, self.jobs_by_type[2].shape[0]))
        self.raytrace_per_type[2] += self.jobs_by_type[2].shape[0]
        self.insersect_triangles()

        if draw_jobs:
            self.draw_jobs()


    def intersect_bounding_box(self):
        jobs = self.jobs_by_type[1]
        if jobs.shape[0] <= 0:
            return
        node_box = self.index.database[1][jobs['node']]
        ray_box = jobs['ray_box']
        ray_box_mask = numpy.all(numpy.logical_and(ray_box[:, 1] >= node_box[:, 0], node_box[:, 1] >= ray_box[:, 0]), axis=-1)
        bounding_box = numpy.stack([numpy.maximum(ray_box[ray_box_mask][:, 0], node_box[ray_box_mask][:, 0]), numpy.minimum(ray_box[ray_box_mask][:, 1], node_box[ray_box_mask][:, 1])], axis=1)
        # Let R be the ray_box
        # Let N be the current Node bounding box
        # We have for each dimension x, y, z, two values: the minimum Vn and the maximum Vx for V in {R, N}, can be arranged as follows
        # Rn --- Rx     Nn --- Nx
        # Rn --- Nn +++ Rx --- Nx
        # Rn --- Nn +++ Nx --- Rx
        # Nn --- Rn +++ Nx --- Rx
        # Nn --- Rn +++ Rx --- Nx
        # Nn --- Nx     Rn --- Rx
        self.logger.log('[raytracer]: level=%d: ray_box_filter(bounding_boxes=%d)' % (self.depth, len(bounding_box)))
        ray_distance = (bounding_box - jobs[ray_box_mask]['ray_origin'][:, None]) / jobs[ray_box_mask]['ray_direction'][:, None]
        box_ray_points = ray_distance[:, :, :, None] * jobs[ray_box_mask]['ray_direction'][:, None, None] + jobs[ray_box_mask]['ray_origin'][:, None, None]
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
        
        has_intersection_index = numpy.where(ray_box_mask)[0][has_intersection]
        jobs['normal'][has_intersection_index] = 0.0
        normal_index = box_ray_distance.argmin(axis=-1)
        jobs['normal'][has_intersection_index, normal_index % 3] = numpy.float32(normal_index // 3) * 2 - 1
        # range_mask = numpy.logical_or(jobs['ray_range'][has_intersection][:, 1] < box_ray_min.data, jobs['ray_range'][has_intersection][:, 0] > box_ray_max.data)
        # print('range_mask.count = %d' % numpy.count_nonzero(range_mask))
        # has_intersection[has_intersection][range_mask] = False
        # jobs['ray_range'][has_intersection] = numpy.stack([
        #     numpy.maximum(box_ray_min.data[numpy.logical_not(range_mask)], jobs['ray_range'][has_intersection][:, 0]),
        #     numpy.minimum(box_ray_max.data[numpy.logical_not(range_mask)], jobs['ray_range'][has_intersection][:, 1]),
        # ], axis=-1)
        ray_range = numpy.stack([
            numpy.maximum(box_ray_min.data, jobs['ray_range'][has_intersection_index][:, 0]) - epsilon,
            numpy.minimum(box_ray_max.data, jobs['ray_range'][has_intersection_index][:, 1]) + epsilon
        ], axis=-1)
        ray_point = jobs[has_intersection_index]['ray_origin'][:, None] + ray_range[:, :, None] * jobs[has_intersection_index]['ray_direction'][:, None]
        ray_box = numpy.stack([ray_point.min(axis=1), ray_point.max(axis=1)], axis=1)
        jobs['ray_range'][has_intersection_index] = ray_range
        jobs['ray_box'][has_intersection_index] = ray_box
        jobs['flags'] = 0
        jobs['flags'][has_intersection_index] = 1
        self.logger.log('[raytracer]: level=%d: raytrace(intersected=%d)' % (self.depth, len(has_intersection_index)))
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
        triangles = self.index.database[2][self.index.nodes[jobs['node']]['ref']]['position']
        triangle_edges = triangles[:, [1, 2, 0]] - triangles
        triangle_normal = normalize(numpy.cross(triangle_edges[:, 2], triangle_edges[:, 0]))
        # triangle_normal_degenerate_mask = numpy.abs(numpy.repeat(numpy.float32(1.0), 3) / triangle_normal) < epsilon
        plane_distance = (numpy.einsum('ab,ab->a', triangle_normal, triangles[:, 0]) - numpy.einsum('ab,ab->a', triangle_normal, jobs['ray_origin'])) / numpy.einsum('ab,ab->a', triangle_normal, jobs['ray_direction'])
        ray_range_mask = numpy.logical_and(plane_distance >= jobs['ray_range'][:, 0], plane_distance <= jobs['ray_range'][:, 1])
        plane_point = jobs['ray_origin'] + plane_distance[:, None] * jobs['ray_direction']
        triangle_inner = plane_point[:, None] - triangles
        barycentric = numpy.stack([numpy.einsum('ab,ab->a', triangle_normal, numpy.cross(triangle_edges[:, i], triangle_inner[:, i])) for i in range(3)], axis=-1)
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
        self.logger.log('[raytracer] Drawing %d primitives...' % (active_jobs.shape[0]))
        pixel_distance = pandas.DataFrame({'y': active_jobs['pixel'][:, 0], 'x': active_jobs['pixel'][:, 1], 'd': active_jobs['ray_range'][:, 0]})
        pixel_depth_resolver = pixel_distance.groupby(['y', 'x'])['d'].idxmin()
        depth_mask = active_jobs['ray_range'][pixel_depth_resolver.values, 0] <= self.depth_buffer[pixel_depth_resolver.index.get_level_values(0).values, pixel_depth_resolver.index.get_level_values(1).values]
        pixel_jobs = active_jobs[pixel_depth_resolver.values][depth_mask]
        # self.color_buffer[active_jobs['pixel'][:, 0], active_jobs['pixel'][:, 1]] = active_jobs['normal'] * 0.5 + 0.5
        self.color_buffer[pixel_jobs['pixel'][:, 0], pixel_jobs['pixel'][:, 1]] = pixel_jobs['normal'] * 0.5 + 0.5
        # self.depth_buffer[active_jobs['pixel'][:, 0], active_jobs['pixel'][:, 1]] = active_jobs['ray_range'][:, 0]
        self.depth_buffer[pixel_jobs['pixel'][:, 0], pixel_jobs['pixel'][:, 1]] = pixel_jobs['ray_range'][:, 0]
        pass

    def schedule_children(self):
        next_jobs = []
        # jobs = numpy.hstack(list(self.jobs_by_type.values()))
        # numpy.argwhere(numpy.logical_and(jobs['pixel'][:, 0] == 240, jobs['pixel'][:, 1] == 440))
        # pixel 440, 240 must be present
        active_job_mask = (self.jobs_by_type[1]['flags'] & 1).astype(bool)
        active_jobs = self.jobs_by_type[1][active_job_mask]
        # while loop
        while True:
            i = len(next_jobs)
            s = 'next_sibling' if i > 0 else 'first_child'
            active_job_nodes = self.index.nodes[active_jobs['node']]
            node_mask = active_job_nodes[s] > 0
            if not numpy.any(node_mask):
                break
            next_jobs.append(active_jobs[node_mask])
            next_jobs[i]['node'] = self.index.nodes[next_jobs[i]['node']][s]
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
            type_mask = self.index.nodes[jobs['node']]['type'] == t
            new_jobs_by_type = jobs[type_mask]
            if len(new_jobs_by_type) > 0:
                self.jobs_by_type[t] = numpy.hstack([self.jobs_by_type[t], new_jobs_by_type])
        pass

    def draw_bounding_boxes(self):
        active_job_mask = (self.jobs_by_type[1]['flags'] & 1).astype(bool)
        active_jobs = self.jobs_by_type[1][active_job_mask]
        box_ray_distance = active_jobs['ray_range'][:, 0]

        # This is only necessary for python, as pandas is more effective than a loop. In GPU, depth-buffer comparison happens in parallel.
        # Note: it might need atomics, but atomics are only integers, depth is float.
        pixel_distance = pandas.DataFrame({'y': active_jobs['pixel'][:, 0], 'x': active_jobs['pixel'][:, 1], 'd': box_ray_distance})
        pixel_depth_resolver = pixel_distance.groupby(['y', 'x'])['d'].idxmin()
        depth_mask = box_ray_distance[pixel_depth_resolver.values] <= self.depth_buffer[pixel_depth_resolver.index.get_level_values(0).values, pixel_depth_resolver.index.get_level_values(1).values]
        pixel_jobs = active_jobs[pixel_depth_resolver.values][depth_mask]
        self.color_buffer[pixel_jobs['pixel'][:, 0], pixel_jobs['pixel'][:, 1]] = pixel_jobs['normal'] * 0.5 + 0.5
        self.depth_buffer[pixel_jobs['pixel'][:, 0], pixel_jobs['pixel'][:, 1]] = box_ray_distance[pixel_depth_resolver.values][depth_mask]

    def display_color_buffer(self):
        color_buffer = (numpy.clip(self.color_buffer, 0.0, 1.0) * 255.0).astype(numpy.uint8)
        image = Image.fromarray(color_buffer)
        image.show()

def parse_index_file(input, error):
    signature = input.read(4)
    if signature != b'\x7fd3i':
        error('Not valid input file: invalid file signature')
    byteorder = 'little'
    version_major = int.from_bytes(input.read(2), byteorder)
    version_minor = int.from_bytes(input.read(2), byteorder)
    if version_major != 0 or version_minor != 1:
        error('Not valid input file: cannot operate on version %d.%d, expected version 0.1' % (version_major, version_minor))
    tree_node_count = int.from_bytes(input.read(4), byteorder)
    tree_depth = int.from_bytes(input.read(4), byteorder)
    tree_root_node = int.from_bytes(input.read(4), byteorder)
    tree_nbytes = int.from_bytes(input.read(4), byteorder)

    database_count = int.from_bytes(input.read(4), byteorder)
    if database_count != 2:
        error('Not valid input file: d3i version 0.1 should only have 2 databases')
    database_size = [0] * database_count
    database_nbytes = [0] * database_count
    for dbi in range(database_count):
        database_size[dbi] = int.from_bytes(input.read(4), byteorder)
        database_nbytes[dbi] = int.from_bytes(input.read(4), byteorder)

    if tree_node_count * MeshIndex.node_dtype.itemsize != tree_nbytes:
        error('Not valid input file: tree_node_count * tree_node_nbytes must be tree_nbytes')
    if database_size[0] * 24 != database_nbytes[0]:
        error('Not valid input file: bounding_box_count * 2 * 3 * sizeof(float32) must be database_nbytes[0]')
    if database_size[1] * 3 * Mesh.vertex_dtype.itemsize != database_nbytes[1]:
        error('Not valid input file: triangle_count * 3 * vertex_nbytes must be database_nbytes[1]')

    buffer_nodes = input.read(tree_nbytes)
    buffer_databases = []
    for dbi in range(database_count):
        buffer_databases.append(input.read(database_nbytes[dbi]))

    input_nodes = numpy.frombuffer(buffer_nodes, dtype=MeshIndex.node_dtype)
    input_database = [None] * 3
    input_database[1] = numpy.frombuffer(buffer_databases[0], dtype=numpy.float32).reshape(-1, 2, 3)
    input_database[2] = numpy.frombuffer(buffer_databases[1], dtype=(Mesh.vertex_dtype, 3))

    return Data(
        depth=tree_depth,
        root_node=tree_root_node,
        nodes=input_nodes,
        database=input_database
    )

def main():
    from argparse import ArgumentParser, FileType
    from pathlib import Path
    from logger import StderrLogger
    numpy.set_printoptions(suppress=True)
    logger = StderrLogger()
    parser = ArgumentParser(description='Raytrace an index file generated by index.py')
    parser.add_argument('--camera-position', type=float, nargs=3, metavar=('x', 'y', 'z'), default=[0.0, 0.0, 0.0], help='The position of the camera.')
    camera_direction_group = parser.add_mutually_exclusive_group()
    camera_direction_group.add_argument('--camera-direction', type=float, nargs=3, metavar=('x', 'y', 'z'), help='The direction vector of the camera.', default=[0.0, 1.0, 0.0])
    camera_direction_group.add_argument('--camera-focus-at', type=float, nargs=3, metavar=('x', 'y', 'z'), help='A point at which the camera is centered upon.')
    parser.add_argument('--world-up', type=float, nargs=3, metavar=('x', 'y', 'z'), default=[0.0, 0.0, 1.0], help='A vector pointing in "up" direction in the world.')
    parser.add_argument('--field-of-view', '--fov', type=float, help='The diagonal field-of-view angle in degrees.')
    parser.add_argument('--input', '-i', required=True, type=lambda value: sys.stdin.buffer if value == '-' else open(Path(value).resolve(strict=True), 'rb'), default=None, help='Index file to raytrace.')
    parser.add_argument('--max-depth', type=int, default=None, help='Cascade on the tree up to maximum depth')
    raytrace_method = parser.add_subparsers(title='method', required=True, dest='method', help='Raytracing method to perform.')
    raytrace_screen = raytrace_method.add_parser('screen', help='Raytrace an image with specified width and height.', description='Raytrace an entire image and store the result as PNG image. If the result is only displayed, it is stored as temporary file.')
    raytrace_screen.add_argument('width', type=int, help='The width of the screen/image')
    raytrace_screen.add_argument('height', type=int, help='The height of the screen/image')
    raytrace_screen.add_argument('--output', '-o', type=lambda value: sys.stdout.buffer if value == '-' else open_file_for_writing(Path(value).resolve(strict=False)), default=None, help='The output file to write a PNG image.')
    raytrace_screen.add_argument('--display', '-d', action='store_true', default=False, help='Display the raytraced image')
    raytrace_pixel = raytrace_method.add_parser('pixel', help='Raytrace a single pixel at (x, y) from image with specified width and height.', description='Raytrace a single pixel and display the result of intermediates for debugging.')
    raytrace_pixel.add_argument('x', type=int, help='The x coordinate of the pixel')
    raytrace_pixel.add_argument('y', type=int, help='The x coordinate of the pixel')
    raytrace_pixel.add_argument('width', type=int, help='The width of the image')
    raytrace_pixel.add_argument('height', type=int, help='The height of the image')
    args = parser.parse_args()

    input_data = parse_index_file(args.input, parser.error)
    args.input.close()

    raytracer_kwargs = {}
    for name in ['aspect_ratio', 'field_of_view', 'camera_position', 'world_up', 'camera_direction', 'camera_focus_at', 'max_depth']:
        if name in args and getattr(args, name) is not None:
            raytracer_kwargs[name] = getattr(args, name)
    raytracer = Raytracer(input_data, logger=logger)
    if args.method == 'screen':
        method = raytracer.screen
        for name in ['width', 'height']:
            if name in args and getattr(args, name) is not None:
                raytracer_kwargs[name] = getattr(args, name)
    elif args.method == 'pixel':
        method = raytracer.pixel
        for name in ['x', 'y', 'width', 'height']:
            if name in args and getattr(args, name) is not None:
                raytracer_kwargs[name] = getattr(args, name)
    else:
        parser.error('Invalid of missing method: %r' % (args.method if 'method' in args else None))
    if 'camera_focus_at' in raytracer_kwargs and 'camera_direction' in raytracer_kwargs:
        del raytracer_kwargs['camera_direction']
    
    method(**raytracer_kwargs)

    if args.method == 'screen':
        if args.output is not None or args.display:
            color_buffer = (numpy.clip(raytracer.color_buffer, 0.0, 1.0) * 255.0).astype(numpy.uint8)
            image = Image.fromarray(color_buffer)
            if args.output is not None:
                image.save(args.output, 'png', optimize=True, compress_level=9)
            if args.display:
                image.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())
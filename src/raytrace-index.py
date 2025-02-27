import sys, os, numpy, pandas, logging
from argparse import ArgumentParser
from pathlib import Path
from util import human_readable_bytes, ByteCounterStream
from mesh import Mesh
from util import normalize
from PIL import Image

logger = logging.getLogger(__name__)

class Raytracer:
    commands = {'image'}

    def __init__(self, args):
        self.input = input
        self.parse_camera_args(args)

    def parse_camera_args(self, args):
        for name in ['width', 'height']:
            value = getattr(args, name)
            if not isinstance(value, int) or value <= 0:
                raise RuntimeError('Expected %s to be a positive integer' % name)
        self.width = args.width
        self.height = args.height
        self.aspect_ratio = args.width / args.height
        if not isinstance(args.field_of_view, float) or args.field_of_view <= 0.0 or args.field_of_view >= 180.0:
            raise RuntimeError('Expected field-of-view to be a positive float in range (0.0; 180.0)')
        self.field_of_view = numpy.radians(args.field_of_view)
        camera_position = numpy.array(args.camera_position, dtype=numpy.float32)
        if len(camera_position.shape) != 1 or camera_position.shape[0] != 3:
            raise ValueError('Option "camera_position" must be 3D vector')
        self.camera_position = camera_position

        world_yaw_direction = -1.0 if args.world[0].isupper() else 1.0
        world_pitch_direction = -1.0 if args.world[1].isupper() else 1.0
        world_roll_direction = -1.0 if args.world[2].isupper() else 1.0
        world_yaw_index = {v: i for i, v in enumerate('xyz')}[args.world[0].lower()]
        world_pitch_index = {v: i for i, v in enumerate('xyz')}[args.world[1].lower()]
        world_roll_index = {v: i for i, v in enumerate('xyz')}[args.world[2].lower()]

        camera_direction = None
        camera_focus_at = None

        if args.camera_direction is None and args.camera_focus_at is None:
            camera_focus_at = numpy.zeros((3,), dtype=numpy.float32)
        if args.camera_focus_at is not None and args.camera_direction is not None:
            raise ValueError('Conflicting options "camera_direction" and "camera_focus_at"')
        if args.camera_focus_at is not None:
            camera_focus_at = numpy.array(args.camera_focus_at, dtype=numpy.float32)
        if camera_focus_at is not None:
            if len(camera_focus_at.shape) != 1 or camera_focus_at.shape[0] != 3:
                raise ValueError('Option "camera_focus_at" must be 3D vector')
            camera_direction = camera_focus_at - camera_position
        if args.camera_direction is not None:
            camera_direction = numpy.array(args.camera_direction, dtype=numpy.float32)
            if len(camera_direction.shape) != 1 or camera_direction.shape[0] != 3:
                raise ValueError('Option "camera_direction" must be 3D vector')
        if numpy.linalg.norm(camera_direction) < numpy.finfo(numpy.float32).eps:
            camera_direction = numpy.zeros((3,), dtype=numpy.float32)
            camera_direction[world_roll_index] = world_roll_direction
        self.camera_direction = normalize(camera_direction)

        self.yaw = numpy.arctan2(world_yaw_direction * self.camera_direction[world_yaw_index], world_roll_direction * self.camera_direction[world_roll_index])
        self.pitch = numpy.arcsin(world_pitch_direction * self.camera_direction[world_pitch_index])

        world_yaw_axis = numpy.zeros((3,), dtype=numpy.float32)
        world_pitch_axis = numpy.zeros((3,), dtype=numpy.float32)
        world_yaw_axis[world_yaw_index] = world_yaw_direction
        world_pitch_axis[world_pitch_index] = world_pitch_direction

        view_yaw = numpy.cross(self.camera_direction, world_yaw_axis)
        view_pitch_length = numpy.linalg.norm(view_yaw)
        if view_pitch_length < numpy.finfo(numpy.float32).eps:
            view_pitch = normalize(numpy.cross(world_pitch_axis, self.camera_direction))
            view_yaw = normalize(numpy.cross(self.camera_direction, view_pitch))
        else:
            view_yaw = view_yaw / view_pitch_length
            view_pitch = normalize(numpy.cross(view_yaw, self.camera_direction))
        self.view_pitch = view_pitch
        self.view_yaw = view_yaw

        self.screen_diagonal = numpy.tan(self.field_of_view * 0.5)
        self.screen_height = self.screen_diagonal / numpy.sqrt(1 + self.aspect_ratio * self.aspect_ratio)
        self.screen_width = self.aspect_ratio * self.screen_height

        self.world_screen_x = self.screen_width * self.view_pitch
        self.world_screen_y = self.screen_height * self.view_yaw
        self.world_screen_center = self.camera_position + self.camera_direction

        view_pixels = numpy.moveaxis(numpy.mgrid[0:args.height, 0:args.width], 0, -1).astype(numpy.float32)
        view_pixels[:, :, 1] = (view_pixels[:, :, 1] + 0.5) / args.width
        view_pixels[:, :, 0] = (view_pixels[:, :, 0] + 0.5) / args.height
        view_pixels[:, :, 0] = 1.0 - view_pixels[:, :, 0]
        view_pixels = view_pixels * 2.0 - 1.0
        view_pixels = view_pixels
        world_pixels = self.world_screen_center + view_pixels[:, :, 0, None] * self.world_screen_y + view_pixels[:, :, 1, None] * self.world_screen_x
        self.primary_ray_direction = normalize(world_pixels - camera_position)

    def image(self, args):
        if args.output is not None or args.display:
            color_buffer = self.primary_ray_direction * 0.5 + 0.5
            color_buffer = (numpy.clip(color_buffer, 0.0, 1.0) * 255.0).astype(numpy.uint8)
            image = Image.fromarray(color_buffer)
            if args.output is not None:
                image.save(args.output, 'png', optimize=True, compress_level=9)
            if args.display:
                image.show()

def main():
    parser = ArgumentParser(description='Raytrace object from wavefront .obj file')
    parser.add_argument('--camera-position', type=float, nargs=3, metavar=('x', 'y', 'z'), default=[0.0, 0.0, -1.0], help='The position of the camera.')
    camera_direction_group = parser.add_mutually_exclusive_group()
    camera_direction_group.add_argument('--camera-direction', type=float, nargs=3, metavar=('x', 'y', 'z'), help='The direction vector of the camera.', default=None)
    camera_direction_group.add_argument('--camera-focus-at', type=float, nargs=3, metavar=('x', 'y', 'z'), help='A point at which the camera is centered upon.')
    parser.add_argument('--world', '-W', choices=['xyz', 'xzy', 'xyZ'], default='xyz', help='The world coordinate system pitch yaw (vertical), pitch (horizontal), roll (forward) axes: <uppercase> = -1')
    parser.add_argument('--field-of-view', '--fov', type=float, default=90.0, help='The diagonal field-of-view angle in degrees.')
    parser.add_argument('--input', '-i', required=True, type=lambda value: sys.stdin if value == '-' else open(Path(value).resolve(strict=True), 'r'), default=None, help='Wavefront .obj file to raytrace.')

    subparsers = parser.add_subparsers(title='command', required=True, dest='command', help='Command to execute.')

    image_parser = subparsers.add_parser('image', help='Raytrace an image with specified width and height.')
    image_parser.add_argument('width', type=int, help='The width of the screen/image')
    image_parser.add_argument('height', type=int, help='The height of the screen/image')
    image_parser.add_argument('--output', '-o', type=lambda value: sys.stdout.buffer if value == '-' else open(Path(value).resolve(strict=False), 'wb'), default=None, help='The output file to write a PNG image.')
    image_parser.add_argument('--display', '-d', action='store_true', default=False, help='Display the raytraced image')

    # pixel_parser = subparsers.add_parser('pixel', help='Raytrace a single pixel at (x, y) from image with specified width and height.')
    # pixel_parser.add_argument('x', type=int, help='The x coordinate of the pixel')
    # pixel_parser.add_argument('y', type=int, help='The y coordinate of the pixel')
    # pixel_parser.add_argument('width', type=int, help='The width of the image')
    # pixel_parser.add_argument('height', type=int, help='The height of the image')

    args = parser.parse_args()

    for name in ['width', 'height']:
        value = getattr(args, name)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError('Expected %s to be a positive integer' % name)

    input_stream = ByteCounterStream(args.input)

    logger.info('[stdin]: read/parse wavefront .obj file...')
    mesh = Mesh.from_wavefront_stream(input_stream)
    args.input.close()
    logger.info(
        '[stdin]: done(bytes_read=%s, triangles=%s)' % (
            human_readable_bytes(input_stream.bytes_read),
            len(mesh.triangles[1:])
        )
    )
    args.input = mesh

    raytracer = Raytracer(args)

    if args.command not in raytracer.commands:
        raise RuntimeError('Unknown command: %s' % args.command)
    getattr(raytracer, args.command)(args)
    return  0

if __name__ == '__main__':
    logging_level = logging.WARNING
    if __debug__:
        logging_level = logging.DEBUG
        numpy.set_printoptions(suppress=True)
    input_logging_level = os.getenv('LOG_LEVEL')
    if input_logging_level is not None and len(input_logging_level) > 0:
        level_names = logging.getLevelNamesMapping()
        if input_logging_level in level_names:
            logging_level = level_names[input_logging_level]
    argparser = ArgumentParser(description='Raytrace object from wavefront .obj file')
    logging.basicConfig(level=logging_level)
    sys.exit(main())
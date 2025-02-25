import sys, os, numpy, pandas, logging
from argparse import ArgumentParser
from pathlib import Path
from util import human_readable_bytes, ByteCounterStream
from mesh import Mesh
from util import normalize

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser(description='Raytrace object from wavefront .obj file')
    parser.add_argument('--camera-position', type=float, nargs=3, metavar=('x', 'y', 'z'), default=[0.0, 0.0, 0.0], help='The position of the camera.')
    camera_direction_group = parser.add_mutually_exclusive_group()
    camera_direction_group.add_argument('--camera-direction', type=float, nargs=3, metavar=('x', 'y', 'z'), help='The direction vector of the camera.', default=[0.0, 1.0, 0.0])
    camera_direction_group.add_argument('--camera-focus-at', type=float, nargs=3, metavar=('x', 'y', 'z'), help='A point at which the camera is centered upon.')
    parser.add_argument('--world-up', type=float, nargs=3, metavar=('x', 'y', 'z'), default=[0.0, 0.0, 1.0], help='A vector pointing in "up" direction in the world.')
    parser.add_argument('--field-of-view', '--fov', type=float, default=90.0, help='The diagonal field-of-view angle in degrees.')
    parser.add_argument('--input', '-i', required=True, type=lambda value: sys.stdin if value == '-' else open(Path(value).resolve(strict=True), 'r'), default=None, help='Wavefront .obj file to raytrace.')

    subparsers = parser.add_subparsers(title='command', required=True, dest='command', help='Command to execute.')

    image_parser = subparsers.add_parser('image', help='Raytrace an image with specified width and height.')
    image_parser.add_argument('width', type=int, help='The width of the screen/image')
    image_parser.add_argument('height', type=int, help='The height of the screen/image')
    image_parser.add_argument('--output', '-o', type=lambda value: sys.stdout.buffer if value == '-' else open(Path(value).resolve(strict=False), 'wb'), default=None, help='The output file to write a PNG image.')
    image_parser.add_argument('--display', '-d', action='store_true', default=False, help='Display the raytraced image')

    pixel_parser = subparsers.add_parser('pixel', help='Raytrace a single pixel at (x, y) from image with specified width and height.')
    pixel_parser.add_argument('x', type=int, help='The x coordinate of the pixel')
    pixel_parser.add_argument('y', type=int, help='The y coordinate of the pixel')
    pixel_parser.add_argument('width', type=int, help='The width of the image')
    pixel_parser.add_argument('height', type=int, help='The height of the image')

    args = parser.parse_args()

    print(args)

    input_stream = ByteCounterStream(args.input)

    logger.info('[stdin]: read/parse wavefront .obj file...')
    mesh = Mesh.from_wavefront_stream(input_stream)
    logger.info(
        '[stdin]: done(bytes_read=%s, triangles=%s)' % (
            human_readable_bytes(input_stream.bytes_read),
            len(mesh.triangles[1:])
        )
    )

    if args.command not in commands:
        raise RuntimeError('Unknown command: %s' % args.command)
    commands[args.command](args)
    return  0

def create_rays(args):
    for name in ['width', 'height']:
        value = getattr(args, name)
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError('Expected %s to be a positive integer' % name)
    if not isinstance(args.field_of_view, float) or args.field_of_view <= 0.0 or args.field_of_view >= 180.0:
        raise RuntimeError('Expected field-of-view to be a positive float in range (0.0; 180.0)')
    aspect_ratio = args.width / args.height
    camera_position = numpy.array(args.camera_position, dtype=numpy.float32)
    if len(camera_position.shape) != 1 or camera_position.shape[0] != 3:
        raise ValueError('Option "camera_position" must be 3D vector')
    if args.camera_direction is None and args.camera_focus_at is None:
        camera_focus_at = numpy.zeros((3,), dtype=numpy.float32)
    if args.camera_focus_at is not None and args.camera_direction is not None:
        raise ValueError('Conflicting options "camera_direction" and "camera_focus_at"')
    if args.camera_focus_at is not None:
        camera_focus_at = numpy.array(args.camera_focus_at, dtype=numpy.float32)
        if len(camera_focus_at.shape) != 1 or camera_focus_at.shape[0] != 3:
            raise ValueError('Option "camera_focus_at" must be 3D vector')
        camera_direction = camera_focus_at - camera_position
    if args.camera_direction is not None:
        camera_direction = numpy.array(args.camera_direction, dtype=numpy.float32)
        if len(camera_direction.shape) != 1 or camera_direction.shape[0] != 3:
            raise ValueError('Option "camera_direction" must be 3D vector')
        camera_direction = normalize(camera_direction)
    world_up = numpy.array(args.world_up, dtype=numpy.float32)
    if len(world_up.shape) != 1 or world_up.shape[0] != 3:
        raise ValueError('Option "world_up" must be 3D vector')
    
    screen_diagonal = numpy.tan(numpy.radians(args.field_of_view) * 0.5)
    screen_height = screen_diagonal / numpy.sqrt(1 + aspect_ratio * aspect_ratio)
    screen_width = aspect_ratio * screen_height

    world_up = normalize(world_up)
    view_right = normalize(numpy.cross(camera_direction, world_up))
    view_up = normalize(numpy.cross(view_right, camera_direction))

    world_screen_up = screen_height * view_up
    world_screen_right = screen_width * view_right
    world_screen_center = camera_position + camera_direction

    view_pixels = numpy.moveaxis(numpy.mgrid[0:args.height, 0:args.width], 0, -1).astype(numpy.float32)
    view_pixels[:, :, 1] = (view_pixels[:, :, 1] + 0.5) / args.width
    view_pixels[:, :, 0] = (view_pixels[:, :, 0] + 0.5) / args.height
    view_pixels[:, :, 0] = 1.0 - view_pixels[:, :, 0]
    view_pixels = view_pixels * 2.0 - 1.0
    world_pixels = world_screen_center + view_pixels[:, :, 0, None] * world_screen_up + view_pixels[:, :, 1, None] * world_screen_right
    ray_direction = normalize(world_pixels - camera_position)
    print(ray_direction)

def command_image(args):
    create_rays(args)

def command_pixel(args):
    pass

commands = {
    'image': command_image,
    'pixel': command_pixel
}

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
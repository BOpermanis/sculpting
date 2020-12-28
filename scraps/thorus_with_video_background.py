import pyglet
import numpy as np
from pyglet.window import key
import ratcave as rc
from OpenGL.GL import *
from OpenGL.GLUT import *
from pprint import pprint
import cv2
import multiprocessing as mp

# Create Window and Add Keyboard State Handler to it's Event Loop
window = pyglet.window.Window()
keys = key.KeyStateHandler()
window.push_handlers(keys)

# Insert filename into WavefrontReader.
obj_filename = rc.resources.obj_primitives
obj_reader = rc.WavefrontReader(obj_filename)

# Create Mesh
torus = obj_reader.get_mesh("Torus", position=(-1, 0, -1.5), scale=.4)
# monkey = obj_reader.get_mesh("Monkey", dynamic=True)
verts = torus.vertices
# print(verts)
# exit()
# Create Scene
scene = rc.Scene(meshes=[torus])
scene.bgColor = 1, 0, 0


def worker(frame_queue):
    # Configure depth and color streams
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def get_frame():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())

    while True:
        frame = frame_queue.get()
        rgb, depth = get_frame()
        mask = np.logical_and(frame[:, :, 2] == 255, frame[:, :, 1] == 0)
        frame[mask] = rgb[mask]
        cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) == 27:
            break

frame_queue = mp.Queue()
t = mp.Process(target=worker, args=(frame_queue,))
t.start()
# Functions to Run in Event Loop
def rotate_meshes(dt):
    verts = torus.vertices
    for v in verts:
        v /= 2.0
    # print(verts.shape, type(verts), verts[0] )
    torus.rotation.x += 80 * dt

def export_frame(dt):
    global frame_queue
    im = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = (im.reshape((480, 640, 3)) * 255).astype(np.uint8)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = np.flipud(im)
    frame_queue.put(im)
    # print(im.shape, np.sum(im))

pyglet.clock.schedule(rotate_meshes)
pyglet.clock.schedule(export_frame)


def move_camera(dt):
    camera_speed = 3
    if keys[key.LEFT]:
        scene.camera.position.x -= camera_speed * dt
    if keys[key.RIGHT]:
        scene.camera.position.x += camera_speed * dt
pyglet.clock.schedule(move_camera)


@window.event
def on_draw():
    with rc.default_shader:
        scene.draw()

# pprint(dir(window))

clock = pyglet.clock.get_default()
platform_event_loop = pyglet.app.platform_event_loop
platform_event_loop.start()


pyglet.app.run()
# while True:
#     dt = clock.update_time()
#     redraw_all = clock.call_scheduled_functions(dt)
#     window.event()
#     # window.switch_to()
#     # window.dispatch_event('on_draw')
#     # window.flip()
#     # window._legacy_invalid = False

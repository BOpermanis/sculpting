import pyglet
import numpy as np
from pyglet.window import key
import ratcave as rc
from OpenGL.GL import *
from OpenGL.GLUT import *
from pprint import pprint
import cv2
import multiprocessing as mp
from camera import Frame


def worker_camera(frame_queue):
    from camera import RsCamera
    from experiments.finger_detector import FingerDetector

    camera = RsCamera()
    finger_detector = FingerDetector()

    while True:
        frame = camera.get()
        rgb = frame.rgb
        depth = frame.depth
        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50

        mask, (y1, x1), (y2, x2) = finger_detector.predict(rgb, depth)
        kp_arr = np.array([(x1, y1), (x2, y2)]) # x un y ir otraadaak
        fingers_3d = camera.convert_depth_frame_to_pointcloud(depth, kp_arr)
        # xs, ys, zs = zip(*fingers_3d)
        frame.fingers_3d = fingers_3d
        frame_queue.put(frame)


def worker_visualize(render_queue):

    while True:
        frame = render_queue.get()
        rgb, depth, render = frame.rgb, frame.depth, frame.render
        mask = np.logical_and(render[:, :, 2] == 255, render[:, :, 1] == 0)
        render[mask] = rgb[mask]
        cv2.imshow('video', cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(10) == 27:
            break


def worker_monitor(queues):
    from time import sleep
    while True:
        s = []
        for k, q in queues.items():
            s.append("{}:{}".format(k, q.qsize()))
        print(", ".join(s))
        sleep(5)


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

scene = rc.Scene(meshes=[torus])
scene.bgColor = 1, 0, 0

frame_queue = mp.Queue()
render_queue = mp.Queue()
threads = [
    mp.Process(target=worker_camera, args=(frame_queue,)),
    mp.Process(target=worker_visualize, args=(render_queue,)),
    mp.Process(target=worker_monitor, args=({"render_queue": render_queue, "frame_queue": frame_queue},))
]
for thread in threads:
    thread.start()

# Functions to Run in Event Loop
def rotate_meshes(dt):
    verts = torus.vertices
    for v in verts:
        v /= 2.0
    # print(verts.shape, type(verts), verts[0] )
    torus.rotation.x += 80 * dt

def export_frame(dt):
    global frame_queue, render_queue

    render = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
    render = np.frombuffer(render, np.float32)
    render = (render.reshape((480, 640, 3)) * 255).astype(np.uint8)
    render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
    render = np.flipud(render)
    frame = frame_queue.get()
    frame.render = render
    render_queue.put(frame)


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

clock = pyglet.clock.get_default()
platform_event_loop = pyglet.app.platform_event_loop
platform_event_loop.start()

pyglet.app.run()

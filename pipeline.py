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


def worker_camera(frame_queue, renderer_feedback_queue):
    from camera import RsCamera
    from experiments.finger_detector import FingerDetector
    from utils import setup_KF

    camera = RsCamera()
    finger_detector = FingerDetector()

    # setup_KF()
    f1 = None
    f2 = None
    while True:
        frame = camera.get()
        rgb = frame.rgb
        depth = frame.depth
        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50

        mask, (y1, x1), (y2, x2) = finger_detector.predict(rgb, depth)
        if np.average(mask) < 0.3:
            kp_arr = np.array([(x1, y1), (x2, y2)]) # x un y ir otraadaak
            fingers_3d = camera.convert_depth_frame_to_pointcloud(depth, kp_arr)[1]

            if f1 is None:
                f1 = setup_KF(fingers_3d[0, 0], fingers_3d[0, 1], fingers_3d[0, 2])
                f2 = setup_KF(fingers_3d[1, 0], fingers_3d[1, 1], fingers_3d[1, 2])
            else:
                f1.predict()
                f2.predict()
                f1.update(fingers_3d[0, :])
                f2.update(fingers_3d[1, :])
        elif f1 is not None:
            f1.predict()
            f2.predict()
            f1.update(f1.get_pred()[:3])
            f2.update(f2.get_pred()[:3])

        frame.finger1 = None
        frame.finger2 = None
        if f1 is not None:
            frame.finger1 = f1.get_pred()[:3]
            frame.finger2 = f2.get_pred()[:3]

        if renderer_feedback_queue.qsize() > 0:
            frame_queue.put(frame)
            renderer_feedback_queue.get()

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


# Insert filename into WavefrontReader.
obj_filename = rc.resources.obj_primitives
obj_reader = rc.WavefrontReader(obj_filename)
print(obj_reader.bodies.keys())
# exit()
# Create Mesh
mesh = obj_reader.get_mesh("Sphere", position=(-1, 1, -1.5), scale=.4)

# monkey = obj_reader.get_mesh("Monkey", dynamic=True)
verts = mesh.vertices

scene = rc.Scene(meshes=[mesh])
scene.bgColor = 1, 0, 0
# from camera import RsCamera
# cam = RsCamera()
#
# print(scene.camera)
# pprint(dir(scene.camera))
# print(scene.camera.model_matrix)
# print(scene.camera.normal_matrix)
# print(scene.camera.projection_matrix)
# print(scene.camera.view_matrix_global)
#
# exit()
window = pyglet.window.Window()
keys = key.KeyStateHandler()
window.push_handlers(keys)

frame_queue = mp.Queue()
render_queue = mp.Queue()
renderer_feedback_queue = mp.Queue()
renderer_feedback_queue.put(0)
threads = [
    mp.Process(target=worker_camera, args=(frame_queue, renderer_feedback_queue)),
    mp.Process(target=worker_visualize, args=(render_queue,)),
    mp.Process(target=worker_monitor, args=({"render_queue": render_queue,
                                             "frame_queue": frame_queue},))
]
for thread in threads:
    thread.start()

def deform_mesh(dt):
    global frame_queue, render_queue, renderer_feedback_queue

    frame = frame_queue.get()

    verts = mesh.vertices
    # for v in verts:
    #     v /= 2.0
    # print(verts.shape, type(verts), verts[0] )
    mesh.rotation.x += 80 * dt

    render = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
    render = np.frombuffer(render, np.float32)
    render = (render.reshape((480, 640, 3)) * 255).astype(np.uint8)
    render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
    render = np.flipud(render)

    frame.render = render
    render_queue.put(frame)
    renderer_feedback_queue.put(0)

pyglet.clock.schedule(deform_mesh)

@window.event
def on_draw():
    with rc.default_shader:
        scene.draw()

clock = pyglet.clock.get_default()
platform_event_loop = pyglet.app.platform_event_loop
platform_event_loop.start()

pyglet.app.run()

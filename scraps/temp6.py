import pyglet
import numpy as np
from pyglet.window import key
import ratcave as rc
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from pprint import pprint

# Create Window and Add Keyboard State Handler to it's Event Loop
window = pyglet.window.Window()
keys = key.KeyStateHandler()
window.push_handlers(keys)

# Insert filename into WavefrontReader.
obj_filename = rc.resources.obj_primitives
obj_reader = rc.WavefrontReader(obj_filename)

# Create Mesh
torus = obj_reader.get_mesh("Torus", position=(-1, 0, -1.5), scale=.4)

# Create Scene
scene = rc.Scene(meshes=[torus])
scene.bgColor = 1, 0, 0


# Functions to Run in Event Loop
def rotate_meshes(dt):
    torus.rotation.x += 80 * dt

def export_frame(dt):
    im = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = (im.reshape((480, 640, 3)) * 255).astype(np.uint8)
    print(im.shape, np.sum(im))

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

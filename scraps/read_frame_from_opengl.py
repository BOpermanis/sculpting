from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
# import serial
import os
from PIL import Image
# import threading
import multiprocessing as mp
# from queue import Queue
import numpy as np
import cv2
from time import sleep
# mp.set_start_method('spawn')

def worker(frame_queue):

    while True:
        frame = frame_queue.get()
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break

frame_queue = mp.Queue()
t = mp.Process(target=worker, args=(frame_queue,))
t.start()

# for _ in range(100):
#     frame_queue.put(np.zeros((100, 100, 3), np.uint8))
#     sleep(1)
# exit()
window = 0

# rotation
X_AXIS = 0.0
Y_AXIS = 0.0
Z_AXIS = 0.0

DIRECTION = 1


def InitGL(Width, Height):
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_SMOOTH)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    # gluPerspective(45.0, float(Width) / float(Height), 0.1, 100.0)
    gluPerspective(33.7, 1.3, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_TEXTURE_2D)

texture_background = glGenTextures(1)


def DrawGLScene():
    global X_AXIS, Y_AXIS, Z_AXIS
    global DIRECTION
    global frame_queue
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()

    glTranslatef(0.0, 0.0, -6.0)

    glRotatef(X_AXIS, 1.0, 0.0, 0.0)
    glRotatef(Y_AXIS, 0.0, 1.0, 0.0)
    glRotatef(Z_AXIS, 0.0, 0.0, 1.0)

    # Draw Cube (multiple quads)
    glBegin(GL_QUADS)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(1.0, 1.0, 1.0)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 1.0, 1.0)
    glVertex3f(-1.0, 1.0, -1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glVertex3f(-1.0, -1.0, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 1.0, -1.0)
    glVertex3f(1.0, 1.0, 1.0)
    glVertex3f(1.0, -1.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)

    glEnd()

    X_AXIS = X_AXIS - 0.60
    Z_AXIS = Z_AXIS - 0.60
    im = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
    im = np.frombuffer(im, np.float32)
    im = (im.reshape((480, 640, 3)) * 255).astype(np.uint8)
    frame_queue.put(im)
    # print(im.shape, np.min(im), np.max(im), im.dtype)
    glutSwapBuffers()


def main():
    global window

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(200, 200)

    window = glutCreateWindow('OpenGL Python Cube')

    glutDisplayFunc(DrawGLScene)
    glutIdleFunc(DrawGLScene)
    InitGL(640, 480)
    glutMainLoop()


if __name__ == "__main__":
    main()
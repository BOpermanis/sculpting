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
from utils import from_homo, initialize_transformation_chessboard2mesh, to_homo, generate_chessboard_in_camera_space, pts2d_from_render, get_chess2render_transformation

def worker_camera(frame_queue, renderer_feedback_queue, proj_mat):
    from camera import RsCamera
    from experiments.finger_detector import FingerDetector
    from utils import setup_KF, project_to_camera, plane_from_pts3d

    camera = RsCamera(flag_return_with_features=2)
    finger_detector = FingerDetector()

    # setup_KF()
    f1 = None
    f2 = None
    K = np.eye(4)
    while True:
        frame = camera.get()
        rgb = frame.rgb
        depth = frame.depth
        # depth = (depth / 56).astype(np.uint8)
        # depth[depth != 0] += 50

        plane_fun = None
        if frame.kp_arr is not None:
            if isinstance(frame.cloud_kp, np.ndarray):
                plane_fun = plane_from_pts3d(frame.cloud_kp)
                for kp in frame.kp_arr:
                    cv2.circle(depth, tuple(kp), 3, (0, 255, 0))

        hand_mask, (y1, x1), (y2, x2) = finger_detector.predict(rgb, depth, plane_fun=plane_fun, K=frame.K)
        fingers_3d = None
        if np.average(hand_mask) < 0.3:
            kp_arr = np.array([(x1, y1), (x2, y2)]) # x un y ir otraadaak
            fingers_3d = camera.convert_depth_frame_to_pointcloud(
                finger_detector.depth_avg, kp_arr, h_target=rgb.shape[0], w_target=rgb.shape[1])[1]

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
        frame.hand_mask = None
        if fingers_3d is not None:
            # frame.finger1 = project_to_camera(fingers_3d[0, :3].reshape(1, -1), proj_mat, K)
            # frame.finger2 = project_to_camera(fingers_3d[1, :3].reshape(1, -1), proj_mat, K)
            # frame.finger1 = project_to_camera(f1.get_pred()[:3].reshape(1, -1), proj_mat.T, K)
            # frame.finger2 = project_to_camera(f2.get_pred()[:3].reshape(1, -1), proj_mat.T, K)
            frame.finger1 = fingers_3d[0, :3].reshape(1, -1)
            frame.finger2 = fingers_3d[1, :3].reshape(1, -1)
            # frame.finger1.fill(0.0)
            # frame.finger2.fill(0.0)
            # frame.finger1[0, 2] = -1.5
            # frame.finger2[0, 2] = -1.5
            frame.hand_mask = hand_mask

        if True: #renderer_feedback_queue.qsize() > 0:
            frame_queue.put(frame)
            renderer_feedback_queue.get()

def worker_visualize(render_queue):
    from time import time
    green = None
    def overlay(rgb, mask, green):
        rgb[mask] = rgb[mask] // 2 + green[mask] // 2
    ts = []
    while True:
        start = time()
        frame = render_queue.get()
        continue

        rgb, depth, render = frame.rgb, frame.depth, frame.render
        if green is None:
            green = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.uint8)
            green[:, :, 1] = 255

        # mask = np.logical_and(render[:, :, 2] == 255, render[:, :, 1] == 0)
        # render[mask] = rgb[mask]
        if frame.hand_mask is not None:
            overlay(render, frame.hand_mask, green)

        for finger in (frame.finger1, frame.finger2):
            if finger is None:
                continue
            finger = np.matmul(frame.K, finger.T).T
            finger = from_homo(finger)[0, :]
            # print(finger)
            x, y = finger
            if not np.isnan(x):
                x = int(np.clip(x, 0, 639))
                y = int(np.clip(y, 0, 479))
                cv2.circle(render, (x, y), 1, (255, 0, 0), 10)

        cv2.imshow('video', cv2.cvtColor(render, cv2.COLOR_RGB2BGR))
        ts.append(time() - start)
        if len(ts) > 10:
            ts.pop(0)
            if frame.id % 10 == 0:
                print("{}) fps={}".format(frame.id, 1 / np.average(ts)))
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
mesh = obj_reader.get_mesh("Sphere", position=(0, 0, -1.5), scale=0.01)
# pprint(dir(mesh))
# exit()
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
    mp.Process(target=worker_camera, args=(frame_queue, renderer_feedback_queue, scene.camera.projection_matrix)),
    mp.Process(target=worker_visualize, args=(render_queue,)),
    mp.Process(target=worker_monitor, args=({"render_queue": render_queue,
                                             "frame_queue": frame_queue},))
]
for thread in threads:
    thread.start()


flag_calibrated = False
centers = list(generate_chessboard_in_camera_space())
calibrartion_renders = []
nr_render = 0

T_chess2render = np.eye(4)

def deform_mesh(dt):
    global frame_queue, render_queue, renderer_feedback_queue, flag_calibrated, nr_render, T_chess2render

    frame = frame_queue.get()

    flag_pinch = False
    if frame.finger1 is not None and frame.finger2 is not None:
        if tuple(frame.finger1.flatten()) == tuple(frame.finger2.flatten()):
            flag_pinch = True

    if flag_calibrated:
        if flag_pinch:
            # x0, y0, z0 = mesh.position
            f1 = frame.finger1
            # f2 = from_homo(np.matmul(to_homo(frame.finger2), P))

            x, y, z = f1[0, :]
            print(f1)
            z = - z * 20
            # print(mesh.position, (x, y, z))
            # mesh.position = (x, y, z)
            # verts = mesh.vertices
            # for v in verts:
            #     pass

        render = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
        render = np.frombuffer(render, np.float32)
        render = (render.reshape((480, 640, 3)) * 255).astype(np.uint8)
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
        render = np.flipud(render)

    else:
        flag_ok_chessboard = False
        if isinstance(frame.cloud_kp, np.ndarray):
            flag_ok_chessboard = frame.cloud_kp.shape[0] == 48

        flag_save_render = nr_render > 5 and len(centers) > 0

        if flag_save_render:
            x, y, z = centers.pop(0)
            mesh.position = x, y, z

        render = glReadPixels(0, 0, 640, 480, GL_RGB, GL_FLOAT)
        render = np.frombuffer(render, np.float32)
        render = (render.reshape((480, 640, 3)) * 255).astype(np.uint8)
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB)
        render = np.flipud(render)

        pt = pts2d_from_render(render)
        if len(pt) > 0:
            if np.linalg.norm(np.asarray(pt) - np.asarray((320, 240))) > 5:
                calibrartion_renders.append(render)
                # print("len(calibrartion_renders)", len(calibrartion_renders), pt)

        if isinstance(frame.cloud_kp, np.ndarray):
            flag_ok_chessboard = frame.cloud_kp.shape[0] == 48

        if len(calibrartion_renders) == 48 and flag_ok_chessboard:
            # import pickle
            # with open("/home/slam_data/four_points_render1.pickle", "wb") as conn:
            #     pickle.dump((calibrartion_renders, frame), conn)
            T = get_chess2render_transformation(calibrartion_renders, frame)
            np.copyto(T_chess2render, T)

    nr_render += 1
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

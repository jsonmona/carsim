from __future__ import print_function, division

import os
import os.path
import cv2
import numpy as np


START_POSITION = np.float64([0.62, 0])
CAMERA_DISPLACEMENT = 0.25
WORLD_WIDTH = 2.7
WORLD_HEIGHT = 3.0
RESOLUTION = (200, 200)
VIEWPORT_WIDTH = 0.2
VIEWPORT_HEIGHT = 0.2
SKY_COLOR = (0, 0, 0)


class CarSimulator(object):
    def __init__(self, jitter=False, bg_path=None):
        if bg_path is not None:
            self.background = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        elif os.path.exists('bg.jpg'):
            self.background = cv2.imread('bg.jpg', cv2.IMREAD_COLOR)
        else:
            self.background = np.zeros((40, 27, 3), dtype='uint8')
            self.background += 240
        self.jitter = not not jitter
        self.reset()
    
    def reset(self):
        self.yaw = np.float64(0)
        self.pos = START_POSITION.copy()
    
    def step(self, speed, rot, dt=1/30):
        """Advances simulator one step forward.
        You may want to call render() to get rendered image

        :param speed: Speed of the vehicle in meters per second.
        :param rot: Rotational speed of the vehicle in radians per second.
        :param dt: Delta time to advance. Defaults to 30 fps.
        """

        if self.jitter:
            speed *= np.random.uniform(0.9, 1.1)
            rot += np.random.uniform(-0.02, 0.02)
            rot *= np.random.uniform(0.9, 1.1)

        self.pos += np.array([np.sin(self.yaw), np.cos(self.yaw)]) * speed * dt
        self.yaw += -rot * dt
    
    def render(self):
        tex_h, tex_w, _ = self.background.shape
        viewport_w, viewport_h = RESOLUTION

        forward = np.array([np.sin(self.yaw), np.cos(self.yaw)], dtype='float32')
        side = np.array([forward[1], -forward[0]], dtype='float32')

        camera_base = self.pos + forward * CAMERA_DISPLACEMENT - side * VIEWPORT_WIDTH / 2

        # In world coordinates
        src_point = np.array([
            camera_base + forward * VIEWPORT_HEIGHT,
            camera_base + forward * VIEWPORT_HEIGHT + side * VIEWPORT_WIDTH,
            camera_base,
        ])

        # To texture coordinates
        src_point[:, 0] *= tex_w / WORLD_WIDTH
        src_point[:, 1] *= tex_h / WORLD_HEIGHT
        src_point[:, 1] = tex_h - src_point[:, 1]
        src_point = src_point.astype('float32')
        
        dst_point = np.array([[0, 0], [viewport_w, 0], [0, viewport_h]], dtype='float32')

        matrix = cv2.getAffineTransform(src_point, dst_point)
        dst = cv2.warpAffine(self.background, matrix, RESOLUTION, None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, SKY_COLOR)

        return dst

        
if __name__ == '__main__':
    sim = CarSimulator()
    while True:
        sim.pos[1] += 0.1
        cv2.imshow('image', sim.render())
        cv2.waitKey(1000)
        if cv2.getWindowProperty('image', cv2.WND_PROP_VISIBLE) < 1:
            break

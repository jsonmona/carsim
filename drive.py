# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import cv2
import scipy
import scipy.ndimage
import time


MAX_SPEED = 0.2
MAX_STEER = np.radians(30)


def color_filter(img, key_color):
    key_color = np.asarray(key_color, dtype='float32')
    key_color = np.expand_dims(np.expand_dims(key_color, 0), 0)

    mask = 255 - np.max(img - key_color, axis=-1)
    return np.clip(mask, 0, 255).astype('uint8')


def detect_cross(black_mask, left=True, right=True):
    h, w = black_mask.shape

    left_val = np.mean(black_mask[h-30:, :w//2]) > 255 * 0.6
    right_val = np.mean(black_mask[h-30:, w//2:]) > 255 * 0.6
    
    return (not left or left_val) and (not right or right_val)


def detect_green(lane, green_mask):
    BLOB_SIZE = 20
    LANE_DIST = 1000
    _, mask = cv2.threshold(green_mask, 0, 255, cv2.THRESH_OTSU)

    max_val = -1
    max_x = 0
    max_y = 0

    for y in range(0, mask.shape[0], BLOB_SIZE // 2):
        xlane = np.polyval(lane, y)

        for x in range(0, mask.shape[1], BLOB_SIZE // 2):
            if x + BLOB_SIZE < xlane - LANE_DIST or xlane + LANE_DIST < x:
                continue

            window = mask[y:y+BLOB_SIZE, x:x+BLOB_SIZE]
            curr_val = np.mean(window)
            if max_val < curr_val:
                max_val = curr_val
                max_x = x + BLOB_SIZE // 2
                max_y = y + BLOB_SIZE // 2
    
    if 220 <= max_val:
        return (max_x, max_y)
    else:
        return None


def sliding_window_lane(mask):
    WINDOW_WIDTH = 80
    WINDOW_HEIGHT = 8
    WINDOW_CNT = 20

    h, w = mask.shape

    # Make histogram on bottom half
    hist = np.mean(mask[h//2:], axis=0, dtype='float32')
    hist = scipy.ndimage.gaussian_filter1d(hist, 15, mode='nearest')

    img = np.expand_dims(mask, -1)
    img = np.tile(img, [1, 1, 3])

    points = []

    curr_x = np.argmax(hist)
    for i in range(WINDOW_CNT):
        offset_x = curr_x.astype('int32') - WINDOW_WIDTH // 2
        offset_x = np.clip(offset_x, 0, w - WINDOW_WIDTH)

        xmin = np.clip(offset_x, 0, w)
        xmax = np.clip(offset_x + WINDOW_WIDTH, 0, w)
        ymin = np.clip(h - (i + 1) * WINDOW_HEIGHT, 0, h)
        ymax = np.clip(h - i * WINDOW_HEIGHT, 0, h)

        window = mask[ymin:ymax, xmin:xmax]
        window = np.mean(window, axis=0)
        colored = np.argwhere(window > 127)
        if colored.size == 0:
            continue

        img[ymin:ymax, xmin:xmax, 1:] = 0
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        
        curr_x = offset_x + np.mean(colored)
        points.append([curr_x, (ymin + ymax) / 2])
    
    if len(points) < 10:
        # TODO: What should I do?
        cv2.imshow('sliding', img)
        return

    points = np.asarray(points, dtype='float32')
    poly = np.polyfit(points[:, 1], points[:, 0], 1)

    lane_draw_y = np.arange(h)
    lane_draw_x = np.polyval(poly, lane_draw_y)
    lane_draw_x += 0.5
    lane_draw_x = lane_draw_x.astype('int32')
    for y, x in zip(lane_draw_y, lane_draw_x):
        if 0 <= x and x < w:
            img[y, x] = [0, 0, 255]
    
    cv2.imshow('sliding', img)
    cv2.moveWindow('sliding', 500, 0)

    return poly


def stanley(lane, car_speed, img_h, img_w):
    distance = np.polyval(lane, img_h * 0.9) - (img_w / 2)

    derived_lane = np.polyder(lane)
    lane_angle = np.polyval(derived_lane, img_h * 0.9)

    k = 0.01
    theta_err = lane_angle
    lat_err = -distance * np.cos(lane_angle)
    return theta_err + np.arctan(k * lat_err / car_speed)


class TaskBase(object):
    def __init__(self):
        self.completed = False
    
    def make_next_task(self):
        return None

    def on_camera(self, lane, img, black_mask, green_mask, lane_mask):
        del img
        return None
    
    def next_task(self):
        if self.completed:
            return self.make_next_task()
        else:
            return self


class FirstTask(TaskBase):
    def __init__(self):
        super(FirstTask, self).__init__()
        self.state = 0
        self.timer = 0
        self.last_move = None
    
    def make_next_task(self):
        return SecondTask()

    def on_camera(self, lane, img, black_mask, green_mask, lane_mask):
        now_time = time.time()

        h, w, _ = img.shape
        if self.state == 0:
            # Wait for first cross
            if detect_cross(black_mask):
                print('[Task 1] Waiting for first cross...', self.timer)
                self.timer += 1
                if 2 <= self.timer:
                    # If cross seen for 2 frames...
                    self.timer = 0
                    self.state = 1
            else:
                self.timer = 0
        elif self.state == 1:
            # Wait for cross to disappear
            if not detect_cross(black_mask):
                print('[Task 1] Waiting for cross to disappear...', self.timer)
                self.timer += 1
                if 10 <= self.timer:
                    # If cross not seen for 10 frames...
                    self.timer = 0
                    self.state = 2
            else:
                self.timer = 0
        elif self.state == 2:
            if self.last_move is None:
                self.last_move = now_time
            elif now_time - self.last_move < 2.5:
                return 0, np.radians(30)
            else:
                self.last_move = None
                self.timer = 0
                self.state = 3
            return 0, np.radians(30)
        elif self.state == 3:
            green = detect_green(lane, green_mask)
            if green is None:
                return
            if 170 <= green[1]:
                self.last_move = None
                self.state = 4
                self.timer = 0
        elif self.state == 4:
            if self.last_move is None:
                print('[Task 1] U-turn')
                self.last_move = now_time
            elif now_time - self.last_move < 10:
                return 0, np.radians(30)
            else:
                self.last_move = None
                self.timer = 0
                self.state = 5
            return 0, np.radians(30)
        elif self.state == 5:
            if detect_cross(black_mask):
                print('[Task 1] Waiting for cross...', self.timer)
                self.timer += 1
                if 2 <= self.timer:
                    # If cross seen for 2 frames...
                    self.timer = 0
                    self.state = 6
            else:
                self.timer = 0
        elif self.state == 6:
            if not detect_cross(black_mask):
                print('[Task 1] Waiting for cross to disappear...', self.timer)
                self.timer += 1
                if 10 <= self.timer:
                    # If cross not seen for 10 frames...
                    self.timer = 0
                    self.state = 7
            else:
                self.timer = 0
        elif self.state == 7:
            if self.last_move is None:
                self.last_move = now_time
            elif now_time - self.last_move < 2.5:
                return 0, np.radians(30)
            else:
                self.last_move = None
                self.timer = 0
                self.state = 8
            return 0, np.radians(30)
        else:
            self.completed = True


class SecondTask(TaskBase):
    def __init__(self):
        super(SecondTask, self).__init__()
        self.state = 0
        self.timer = 0
        self.last_move = None
    
    def make_next_task(self):
        return ThirdTask()

    def on_camera(self, lane, img, black_mask, green_mask, lane_mask):
        now_time = time.time()

        if self.state == 0:
            # Wait for first cross
            if detect_cross(black_mask, left=False):
                print('[Task 2] Waiting for right cross...', self.timer)
                self.timer += 1
                if 2 <= self.timer:
                    # If cross seen for 2 frames...
                    self.timer = 0
                    self.state = 1
            else:
                self.timer = 0
        elif self.state == 1:
            # Wait for cross to disappear
            if not detect_cross(black_mask, left=False):
                print('[Task 2] Waiting for cross to disappear...', self.timer)
                self.timer += 1
                if 10 <= self.timer:
                    # If cross not seen for 10 frames...
                    self.last_move = None
                    self.timer = 0
                    self.state = 2
            else:
                self.timer = 0
        elif self.state == 2:
            if self.last_move is None:
                self.last_move = now_time
            elif now_time - self.last_move < 2.5:
                return 0, np.radians(-30)
            else:
                self.last_move = None
                self.timer = 0
                self.state = 3
            return 0, np.radians(-30)
        else:
            self.completed = True


class ThirdTask(TaskBase):
    def __init__(self):
        super(ThirdTask, self).__init__()
        self.state = 0
        self.timer = 0
        self.last_move = 0
    
    def make_next_task(self):
        return None

    def on_camera(self, lane, img, black_mask, green_mask, lane_mask):
        now_time = time.time()

        if self.state <= 1:
            #TODO: Make falling edge detector, not level detector
            if now_time - self.last_move < 5:
                return
            green = detect_green(lane, green_mask)
            print(green)
            if green is None:
                return
            if 170 <= green[1]:
                print('[Task 3] Seen green '+str(self.state)+' times')
                self.last_move = now_time
                self.state += 1
                self.timer = 0
        elif self.state == 2:
            # Wait for first cross
            if detect_cross(black_mask, right=False):
                print('[Task 3] Waiting for left cross...', self.timer)
                self.timer += 1
                if 2 <= self.timer:
                    # If cross seen for 2 frames...
                    self.timer = 0
                    self.state = 3
            else:
                self.timer = 0
        elif self.state == 3:
            # Wait for cross to disappear
            if not detect_cross(black_mask, right=False):
                print('[Task 3] Waiting for cross to disappear...', self.timer)
                self.timer += 1
                if 10 <= self.timer:
                    # If cross not seen for 10 frames...
                    self.timer = 0
                    self.state = 4
            else:
                self.timer = 0
        elif self.state == 4:
            if self.last_move is None:
                self.last_move = now_time
            elif now_time - self.last_move < 2.5:
                return 0, np.radians(30)
            else:
                self.last_move = None
                self.timer = 0
                self.state = 5
            return 0, np.radians(30)
        else:
            self.completed = True


class Drive(object):
    def __init__(self):
        self.do_movement = None
        self.lane = None
        self.task = FirstTask()

    def on_camera(self, img):
        black_mask = color_filter(img, [0, 0, 0])
        green_mask = color_filter(img, [20, 220, 20])

        green_mask -= black_mask
        lane_mask = np.maximum(black_mask, green_mask)
        #cv2.imshow('lane_mask', lane_mask)

        _, lane_mask = cv2.threshold(lane_mask, 0, 255, cv2.THRESH_OTSU)

        new_lane = sliding_window_lane(lane_mask)
        if new_lane is not None:
            self.lane = new_lane
        
        if self.lane is None:
            return
        
        if self.task is not None:
            task_ret = self.task.on_camera(self.lane, img, black_mask, green_mask, lane_mask)
            self.task = self.task.next_task()
        else:
            task_ret = None

        if task_ret is None:
            # Drive by lane
            new_speed = 0.1
            new_rot = stanley(self.lane, 0.1, img.shape[0], img.shape[1])
        else:
            new_speed, new_rot = task_ret

        new_speed = np.clip(new_speed, -MAX_SPEED, MAX_SPEED)
        new_rot = np.clip(new_rot, -MAX_STEER, MAX_STEER)

        if self.do_movement is not None:
            self.do_movement(new_speed, new_rot)


def test_drive():
    from carsim import CarSimulator

    sim = CarSimulator(jitter=True)
    sim.reset()

    drive = Drive()
    drive.do_movement = sim.step

    while True:
        cam = sim.render()
        cv2.imshow('cam', cam)
        cv2.moveWindow('cam', 0, 0)
        cv2.waitKey(33)
        if cv2.getWindowProperty('cam', cv2.WND_PROP_VISIBLE) < 1:
            break

        drive.on_camera(cam)

if __name__ == '__main__':
    np.seterr(all='raise')
    test_drive()

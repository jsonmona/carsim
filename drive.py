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
    cv2.moveWindow('sliding', 400, 0)

    return poly


def stanley(lane, car_speed, img_h, img_w):
    car_speed = max(0.1, car_speed)
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

    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        return None


class TurnLeftTask(TaskBase):
    def __init__(self, least_wait=None):
        super(TurnLeftTask, self).__init__()
        self.timer = None
        self.streak = 0
        self.wait_flip = False
        self.least_wait = 1 if least_wait is None else least_wait

    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        img_h, img_w = img.shape[:2]
        slope = lane[-2]

        if self.timer is None:
            self.timer = time.time()
            self.wait_flip = 0 < slope
        elif time.time() - self.timer < self.least_wait or not has_lane or (self.wait_flip and 0 < slope):
            return 0, MAX_STEER
        else:
            self.wait_flip = False
        
        if 0 < slope and slope < 0.8:
            self.streak += 1
        else:
            self.streak = 0
        
        if 5 <= self.streak:
            self.completed = True
            return
        
        if 0 < slope and slope < 1.5:
            lane_dist = (img_w // 2) - np.polyval(lane, img_h)
            if img_w * 0.4 <= abs(lane_dist):
                return MAX_SPEED / 4, np.radians(5)

        return 0, MAX_STEER


class TurnRightTask(TaskBase):
    def __init__(self):
        super(TurnRightTask, self).__init__()
        self.timer = None
        self.streak = 0
        self.wait_flip = False

    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        img_h, img_w = img.shape[:2]
        slope = lane[-2]

        if self.timer is None:
            self.timer = time.time()
            self.wait_flip = 0 > slope
        elif time.time() - self.timer < 1 or not has_lane or (self.wait_flip and 0 > slope):
            return 0, -MAX_STEER
        else:
            self.wait_flip = False
        
        if 0 > slope and slope > -0.8:
            self.streak += 1
        else:
            self.streak = 0
        
        if 5 <= self.streak:
            self.completed = True
            return
        
        if 0 > slope and slope > -1.5:
            lane_dist = (img_w // 2) - np.polyval(lane, img_h)
            if img_w * 0.4 <= abs(lane_dist):
                return MAX_SPEED / 4, np.radians(5)

        return 0, -MAX_STEER

class WaitCrossTask(TaskBase):
    def __init__(self, left=True, right=True):
        super(WaitCrossTask, self).__init__()
        self.left = left
        self.right = right
    
    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        y_pos = img.shape[0] * 8 // 10

        x_center = np.polyval(lane, y_pos)
        x_center = np.clip(x_center, 1, img.shape[1] - 1).astype('int32')

        crossed = True

        if crossed and self.left:
            crossed = np.mean(lane_mask[y_pos, :x_center]) > 220
        if crossed and self.right:
            crossed = np.mean(lane_mask[y_pos, x_center:]) > 220
        
        if crossed:
            self.completed = True


class SleepTask(TaskBase):
    def __init__(self, sec):
        super(SleepTask, self).__init__()
        self.timer = None
        self.sec = sec
    
    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        if self.timer is None:
            self.timer = time.time()
        elif self.sec <= time.time() - self.timer:
            self.completed = True


class HaltTask(TaskBase):
    def __init__(self, sec=None):
        super(HaltTask, self).__init__()
        self.timer = None
        self.sec = sec
    
    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        if self.timer is None:
            self.timer = time.time()
        elif self.sec is not None and self.sec <= time.time() - self.timer:
            self.completed = True
        
        return 0, 0

class WaitGreenMarkerTask(TaskBase):
    def __init__(self):
        super(WaitGreenMarkerTask, self).__init__()
    
    def on_camera(self, has_lane, lane, img, black_mask, green_mask, lane_mask):
        BLOB_SIZE = 20
        LANE_DIST = 100

        img_h = img.shape[0]
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
        
        if 220 <= max_val and img_h // 2 < max_y:
            self.completed = True


class Drive(object):
    def __init__(self):
        self.do_movement = None
        self.lane = None
        self.curr_speed = 0
        self.curr_rot = 0
        self.task_idx = 0
        self.tasks = [
            WaitCrossTask(),
            SleepTask(0.5),
            TurnLeftTask(),
            WaitGreenMarkerTask(),
            TurnLeftTask(least_wait=3.5),
            WaitCrossTask(),
            TurnLeftTask(),
            SleepTask(5),
            WaitCrossTask(left=False),
            SleepTask(0.5),
            TurnRightTask(),
            WaitGreenMarkerTask(),
            SleepTask(3),
            WaitGreenMarkerTask(),
            WaitCrossTask(right=False),
            SleepTask(0.5),
            TurnLeftTask(),
            WaitGreenMarkerTask(),
            HaltTask(),
        ]

    def on_camera(self, img):
        black_mask = color_filter(img, [0, 0, 0])
        green_mask = color_filter(img, [20, 220, 20])

        green_mask -= black_mask
        lane_mask = np.maximum(black_mask, green_mask)

        #cv2.imshow('lane_mask', lane_mask)
        #cv2.moveWindow('lane_mask', 800, 0)

        found_threshold, lane_mask = cv2.threshold(lane_mask, 0, 255, cv2.THRESH_OTSU)

        # Try detect lane only when there is something dark
        new_lane = None
        if 80 <= found_threshold:
            new_lane = sliding_window_lane(lane_mask)
            if new_lane is not None:
                self.lane = new_lane
            
            if self.lane is None:
                return
        
        if self.task_idx < len(self.tasks):
            now_task = self.tasks[self.task_idx]
            task_ret = now_task.on_camera(new_lane is not None, self.lane, img, black_mask, green_mask, lane_mask)
            if now_task.completed:
                self.task_idx += 1
                if self.task_idx < len(self.tasks):
                    print('Task is now '+repr(self.tasks[self.task_idx]))
                else:
                    print('Task is now None')
        else:
            task_ret = None

        if task_ret is None:
            # Drive by lane
            new_speed = MAX_SPEED
            new_rot = stanley(self.lane, self.curr_speed, img.shape[0], img.shape[1])
        else:
            new_speed, new_rot = task_ret

        new_speed = np.clip(new_speed, -MAX_SPEED, MAX_SPEED)
        new_rot = np.clip(new_rot, -MAX_STEER, MAX_STEER)

        self.curr_speed = self.curr_speed * 0.6 + new_speed * 0.4
        self.curr_rot = self.curr_rot * 0.6 + new_rot * 0.4

        if self.do_movement is not None:
            self.do_movement(self.curr_speed, self.curr_rot)


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

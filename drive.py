# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import cv2
import scipy
import scipy.ndimage
import time


MAX_SPEED = 0.5
MAX_STEER = np.radians(90)

# 차량의 회전 중심점으로부터 Bird's eye view 처리 후 이미지 하단까지의 거리 (픽셀 단위)
# Bird's eye view 처리 후 이미지의 높이가 실제 몇 미터인지 재서 비율을 구하면 됨
DIST_CAM_FROM_MASS = 250


def color_filter(img, key_color):
    key_color = np.asarray(key_color, dtype='float32')
    key_color = np.expand_dims(np.expand_dims(key_color, 0), 0)

    mask = 255 - np.max(img - key_color, axis=-1)
    return np.clip(mask, 0, 255).astype('uint8')


def nms_vectors(vectors, threshold_rad, out_normalized=False):
    """벡터들을 상대로 방향을 기준으로 NMS 연산을 해서 비슷한 벡터들을 없앰"""
    threshold_cos = np.cos(threshold_rad)
    magnitude = np.linalg.norm(vectors, keepdims=True, axis=-1)
    direction = vectors / magnitude

    to_retain = np.ones(len(vectors), dtype='bool')

    for i in np.argsort(magnitude[:, 0])[::-1]:
        if not to_retain[i]:
            continue

        now_dir = direction[i]
        cos_angle = np.dot(direction, now_dir)
        now_retain = cos_angle < threshold_cos
        now_retain[i] = True
        to_retain &= now_retain
    
    if out_normalized:
        return direction[to_retain]
    else:
        return vectors[to_retain]


class SlidingWindowLaneDetector(object):
    def __init__(self):
        pass

    def predict(self, mask):
        MIN_WINDOW_CNT = 10
        WINDOW_WIDTH = 80
        WINDOW_HEIGHT = 8
        WINDOW_CNT = 22

        h, w = mask.shape

        # Make histogram on bottom half
        hist = np.mean(mask[h//2:], axis=0, dtype='float32')
        hist = scipy.ndimage.gaussian_filter1d(hist, 15, mode='nearest')

        img = np.expand_dims(mask, -1)
        img = np.tile(img, [1, 1, 3])

        #mask = mask > 127
        points = []

        delta_x = 0
        curr_x = np.argmax(hist)
        for i in range(WINDOW_CNT):
            offset_x = curr_x.astype('int32') - WINDOW_WIDTH // 2
            offset_x = np.clip(offset_x, 0, w - WINDOW_WIDTH)

            xmin = offset_x
            xmax = xmin + WINDOW_WIDTH
            ymin = h - (i + 1) * WINDOW_HEIGHT
            ymax = h - i * WINDOW_HEIGHT

            window = mask[ymin:ymax, xmin:xmax]
            window = np.mean(window, axis=0)
            colored = np.argwhere(window > 127)
            if colored.size == 0:
                curr_x += delta_x
                continue

            img[ymin:ymax, xmin:xmax, 1:] = 0
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            
            next_x = offset_x + np.mean(colored)
            delta_x = next_x - curr_x
            curr_x = next_x

            points.append([curr_x, (ymin + ymax) / 2])
        
        if len(points) < MIN_WINDOW_CNT:
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
        cv2.moveWindow('sliding', 600, 0)

        return poly


class HoughLaneDetector(object):
    def __init__(self):
        self.frame = 0
        self.last_left_line = None
        self.last_right_line = None
        self.last_middle_line = None
    
    def predict(self, mask):
        # 얼마나 확실하게 왼쪽/오른쪽인 엣지만 걸러낼지
        EDGE_TOLERANT = 20

        img = np.expand_dims(mask, -1)
        img = np.tile(img, [1, 1, 3])

        # TODO: Tune parameter
        edges = cv2.Canny(mask, 70, 120)
        cv2.imshow('edges', edges)
        cv2.moveWindow('edges', 900, 0)

        horizontal_mask = cv2.Sobel(mask, cv2.CV_8U, 1, 0)

        left_edges = np.where(127 - EDGE_TOLERANT <= horizontal_mask, edges, 0)
        right_edges = np.where(horizontal_mask <= 128 + EDGE_TOLERANT, edges, 0)

        left_line, left_line_segments = get_line_proposal(left_edges)
        right_line, right_line_segments = get_line_proposal(right_edges)

        # Predict middle line if not both left and right are present
        middle_line = None

        if left_line is not None and right_line is not None:
            middle_line = (left_line + right_line) / 2
        elif self.last_middle_line is not None:
            curr_line = None
            if left_line is not None and self.last_left_line is not None:
                curr_line = left_line
                prev_line = self.last_left_line
            elif right_line is not None and self.last_right_line is not None:
                curr_line = right_line
                prev_line = self.last_right_line
            if curr_line is not None:
                eval_ys = np.asarray([0, img.shape[0]])
                curr_pts = np.polyval(curr_line, eval_ys)
                prev_pts = np.polyval(prev_line, eval_ys)
                prev_middle_pts = np.polyval(self.last_middle_line, eval_ys)
                curr_middle_pts = prev_middle_pts - prev_pts + curr_pts
                middle_line = np.polyfit(eval_ys, curr_middle_pts, 1)

        # plotting

        img = np.expand_dims(mask, -1)
        img = np.tile(img, [1, 1, 3])

        for xyxy in left_line_segments:
            pt1 = xyxy[0], xyxy[1]
            pt2 = xyxy[2], xyxy[3]
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)

        for xyxy in right_line_segments:
            pt1 = xyxy[0], xyxy[1]
            pt2 = xyxy[2], xyxy[3]
            cv2.line(img, pt1, pt2, (0, 0, 255), 2)

        cv2.imshow('hough', img)
        cv2.moveWindow('hough', 600, 0)

        img = np.expand_dims(mask, -1)
        img = np.tile(img, [1, 1, 3])

        if left_line is not None:
            ys = np.arange(img.shape[0])
            xs = np.polyval(left_line, ys).astype('int32') - 1
            for y, x in zip(ys, xs):
                img[y, x:x+3] = [255, 0, 0]

        if right_line is not None:
            ys = np.arange(img.shape[0])
            xs = np.polyval(right_line, ys).astype('int32') - 1
            for y, x in zip(ys, xs):
                img[y, x:x+3] = [0, 0, 255]

        if middle_line is not None:
            ys = np.arange(img.shape[0])
            xs = np.polyval(middle_line, ys).astype('int32') - 1
            for y, x in zip(ys, xs):
                img[y, x:x+3] = [0, 255, 0]

        cv2.imshow('lane', img)
        cv2.moveWindow('lane', 900, 0)

        self.last_left_line = left_line
        self.last_right_line = right_line
        self.last_middle_line = middle_line

        return middle_line


def get_line_proposal(edge_img):
    """수평 대비 각도가 LEAST_ANGLE이상인 라인들 중,
    제일 긴 라인을 기준으로, 그로부터 각도가 MAX_ANGLE이내인 선분들의 평균을 구함"""
    LEAST_ANGLE = 10
    MAX_ANGLE = 30

    up_vector = np.float32([0, 1])
    least_up_angle = np.cos(np.radians(90 - LEAST_ANGLE))
    max_cos_angle = np.cos(np.radians(MAX_ANGLE))

    # 최소 40px 이어져 있어야 하고, 10px 끊어져 있어도 직선으로 판단
    lines = cv2.HoughLinesP(edge_img, 1, np.pi / 180, 20, None, 40, 10)

    if lines is None:
        return None, np.zeros((0, 4), dtype='int32')
    
    # xyxy순서
    lines = lines[:, 0, :]
    
    # 벡터로 변환 후 방향과 크기로 분해
    good_lines = lines.astype('float32')
    direction = good_lines[:, 2:] - good_lines[:, :2]
    magnitude = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction /= magnitude

    # up vector와 각도 차이를 구함
    # abs를 취해서 뒤집어진 벡터도 고려함
    cos_angles = np.dot(direction, up_vector)
    good_lines = good_lines[least_up_angle < np.abs(cos_angles)]

    if len(good_lines) == 0:
        return None, np.zeros((0, 4), dtype='int32')

    # 필터링 이후 다시 벡터로 변환
    direction = good_lines[:, 2:] - good_lines[:, :2]
    magnitude = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction /= magnitude

    largest_idx = np.argmax(magnitude)

    # 각도 차이로 필터링함 (cos(angle) = dot product)
    # abs를 취해서 뒤집어진 벡터도 고려함
    cos_angles = np.dot(direction, direction[largest_idx])
    good_lines = good_lines[max_cos_angle < np.abs(cos_angles)]

    starts = good_lines[:, :2]
    ends = good_lines[:, 2:]
    lengths = np.linalg.norm(ends - starts, axis=-1)
    
    points = np.reshape(good_lines, [-1, 2])
    weights = np.repeat(lengths / 2, 2)

    poly = np.polyfit(points[:, 1], points[:, 0], 1, w=weights)
    return poly, lines


def stanley(lane, car_speed, img_h, img_w):
    car_speed = max(0.1, car_speed)
    distance = np.polyval(lane, img_h * 0.9) - (img_w / 2)

    derived_lane = np.polyder(lane)
    lane_angle = np.polyval(derived_lane, img_h * 0.9)

    k = 0.01
    theta_err = lane_angle
    lat_err = -distance * np.cos(lane_angle)
    return theta_err + np.arctan(k * lat_err)


class TaskBase(object):
    def __init__(self):
        self.completed = False
    
    def make_next_task(self):
        return None

    def on_camera(self, *args):
        return None


class TurnStillTask(TaskBase):
    def __init__(self, target_deg, direction, finish_on_lane=False):
        super(TurnStillTask, self).__init__()
        self.target_rad = np.radians(target_deg, dtype='float32')
        if direction == 'left':
            self.rot_speed = MAX_STEER
        else:
            self.rot_speed = -MAX_STEER
        self.frames_no_estimate = 2
        self.frames_can_estimate = 7
        self.turning_speed = self.rot_speed
        self.start_time = None
        self.prev_lines = []
        self.prev_time = None
    
    def set_degree(self, target_deg):
        self.target_rad = np.radians(target_deg, dtype='float32')
    
    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        now_time = time.time()

        if self.start_time is None:
            self.start_time = now_time

        if 0 < self.frames_no_estimate:
            self.frames_no_estimate -= 1
        elif 0 < self.frames_can_estimate:
            self.frames_can_estimate -= 1
            if has_lane:
                # detect lines from mask
                # TODO: Tune parameter
                edges = cv2.Canny(lane_mask, 70, 120)

                lines = cv2.HoughLinesP(edges, 4, np.radians(10), 20, None, 40, 10)
                if lines is None:
                    lines = np.empty((0, 4), dtype='float32')
                else:
                    lines = lines[:, 0].astype('float32')

                # Draw lines
                visualize = np.expand_dims(edges, -1)
                visualize = np.tile(visualize, [1, 1, 3])
                for line in lines:
                    pt1 = int(line[0, 0]), int(line[0, 1])
                    pt2 = int(line[0, 2]), int(line[0, 3])
                    cv2.line(edges, pt1, pt2, (0, 255, 0))

                cv2.imshow('turn', edges)
                cv2.moveWindow('turn', 0, 400)

                
                # convert into vector
                lines = lines[:, :2] - lines[:, 2:]
                lines = nms_vectors(lines, np.radians(5), out_normalized=True)

                # estimate rotating speed
                if 0 < len(self.prev_lines) and 0 < len(lines) and now_time - self.prev_time < 999:
                    dt = now_time - self.prev_time

                    similarity = np.dot(lines, self.prev_lines.T)
                    most_similar_curr = np.argmax(similarity)
                    most_similar_prev = most_similar_curr % len(self.prev_lines)
                    most_similar_curr = most_similar_curr // len(self.prev_lines)

                    estimated_speed = np.arccos(similarity[most_similar_curr, most_similar_prev]) / dt

                    # Within seemingly valid range and has correct sign
                    if 0.1 <= np.abs(estimated_speed) <= np.abs(self.rot_speed) * 3 and 0 < estimated_speed * self.rot_speed:
                        self.turning_speed = self.turning_speed * 0.8 + estimated_speed * 0.2
                        print(estimated_speed, self.turning_speed, self.rot_speed)

                # update history
                self.prev_lines = lines
                self.prev_time = now_time
        
        if self.target_rad <= np.abs(self.turning_speed) * (now_time - self.start_time):
            self.completed = True

        return 0, self.rot_speed


class TurnMovingTask(TaskBase):
    def __init__(self, direction, speed=None):
        super(TurnMovingTask, self).__init__()
        if direction == 'left':
            self.rot_speed = MAX_STEER
            self.slope_sign = 1
        else:
            self.rot_speed = -MAX_STEER
            self.slope_sign = -1
        
        self.speed = MAX_SPEED if speed is None else speed
        self.wait_flip = None
        
    
    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        slope = lane[-2] * self.slope_sign

        if self.wait_flip is None:
            self.wait_flip = slope <= 0.2
        elif not has_lane or (self.wait_flip and 0.2 < slope):
            self.wait_flip = False
        
        if not self.wait_flip and 0 <= slope:
            self.completed = True
            return

        return self.speed, self.rot_speed


class SleepTask(TaskBase):
    def __init__(self, sec):
        super(SleepTask, self).__init__()
        self.timer = None
        self.sec = sec
    
    def on_camera(self, *args):
        if self.timer is None:
            self.timer = time.time()
        elif self.sec <= time.time() - self.timer:
            self.completed = True


class HaltTask(TaskBase):
    def __init__(self, sec=None):
        super(HaltTask, self).__init__()
        self.timer = None
        self.sec = sec
    
    def on_camera(self, *args):
        if self.timer is None:
            self.timer = time.time()
        elif self.sec is not None and self.sec <= time.time() - self.timer:
            self.completed = True
        
        return 0, 0

class WaitGreenMarkerTask(TaskBase):
    def __init__(self, far=None):
        super(WaitGreenMarkerTask, self).__init__()
        if far is not None:
            self.far = far
        else:
            self.far = 0.5
    
    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        BLOB_SIZE = 20
        LANE_DIST = 100
        THRESHOLD = 180

        img_h = img.shape[0]
        _, mask = cv2.threshold(green_mask, 0, 255, cv2.THRESH_OTSU)

        max_val = -1
        max_x = 0
        max_y = 0

        for y in range(0, mask.shape[0], BLOB_SIZE // 4):
            xlane = np.polyval(lane, y)

            for x in range(0, mask.shape[1], BLOB_SIZE // 4):
                if x + BLOB_SIZE < xlane - LANE_DIST or xlane + LANE_DIST < x:
                    continue

                window = mask[y:y+BLOB_SIZE, x:x+BLOB_SIZE]
                curr_val = np.mean(window)
                if max_val < curr_val:
                    max_val = curr_val
                    max_x = x + BLOB_SIZE // 2
                    max_y = y + BLOB_SIZE // 2
        
        if THRESHOLD <= max_val and img_h * (1 - self.far) < max_y:
            self.completed = True


class Drive(object):
    def __init__(self):
        self.do_movement = None
        self.lane_detector = SlidingWindowLaneDetector()
        self.lane = None
        self.curr_speed = 0
        self.curr_rot = 0
        self.task_idx = 0
        self.tasks = [
            TurnStillTask(90, 'left'),
            HaltTask(),
            # SleepTask(999999),

            WaitGreenMarkerTask(1.0),
            TurnMovingTask('right'),
            WaitGreenMarkerTask(0.2),
            TurnStillTask(90, 'right'),
            # obj 1
            TurnStillTask(90, 'right', True),

            WaitGreenMarkerTask(0.8),
            TurnMovingTask('right'),
            WaitGreenMarkerTask(),
            SleepTask(1),
            # obj 2
            WaitGreenMarkerTask(1.0),
            TurnStillTask(90, 'left'),
            TurnStillTask(90, 'right', True),
            WaitGreenMarkerTask(),
            # obj 3
            WaitGreenMarkerTask(),
            # obj 4
            TurnMovingTask('left'),
            WaitGreenMarkerTask(),
            HaltTask(),
        ]

    def on_camera(self, img):
        cv2.imshow('img', img)
        cv2.moveWindow('img', 0, 0)

        lane_mask = np.maximum(img[:, :, 0], img[:, :, 2])
        lane_mask = 255 - lane_mask

        green_mask = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(green_mask, (40, 80, 0), (80, 255, 255))
        cv2.imshow('green_mask', green_mask)
        cv2.moveWindow('green_mask', 300, 0)

        new_lane = self.lane_detector.predict(lane_mask)
        if new_lane is not None:
            self.lane = new_lane
        
        if self.lane is None:
            return
        
        if self.task_idx < len(self.tasks):
            now_task = self.tasks[self.task_idx]
            task_ret = now_task.on_camera(new_lane is not None, self.lane, img, lane_mask, green_mask)
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

    sim = CarSimulator(jitter=False, delay=0)
    sim.reset()

    drive = Drive()
    drive.do_movement = sim.step

    while True:
        cam = sim.render()
        drive.on_camera(cam)
        
        if cv2.waitKey(33) & 0xff == ord('q'):
            break
        if cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE) < 1:
            break

if __name__ == '__main__':
    np.seterr(all='raise')
    test_drive()

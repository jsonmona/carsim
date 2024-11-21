# -*- coding: utf-8 -*-
from __future__ import print_function, division

import numpy as np
import cv2
import scipy
import scipy.ndimage
import time


MAX_SPEED = 0.5
MAX_STEER = np.radians(60)

# 시뮬레이터를 위해, 시계를 이만큼 더 느리게 측정함
# 실제 주행시는 1로 설정
CLOCK_SLOWDOWN = 10

# 차량의 회전 중심점으로부터 Bird's eye view 처리 후 이미지 하단까지의 거리 (픽셀 단위)
# Bird's eye view 처리 후 이미지의 높이가 실제 몇 미터인지 재서 비율을 구하면 됨
DIST_CAM_FROM_MASS = 300

# 1px이 몇 미터인지 (meter per pixel)
PIXEL_TO_METER = 0.0001


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
        MIN_WINDOW_CNT = 15
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
    
    def on_start(self, prev_task):
        return

    def on_camera(self, *args):
        return


class TurnStillTask(TaskBase):
    def __init__(self, target_deg, direction, finish_on_lane=False, finish_on_task=None):
        """
        :param bool finish_on_lane: 차선이 보이면 쫑료할지 여부
        :param TaskBase | None finish_on_task: 주어진 task가 completed가 되면 종료
        """
        super(TurnStillTask, self).__init__()
        self.target_rad = np.radians(target_deg, dtype='float32')
        if direction == 'left':
            self.rot_speed = MAX_STEER
        else:
            self.rot_speed = -MAX_STEER
        self.finish_on_lane = finish_on_lane
        self.finish_on_task = finish_on_task
        self.frames_no_estimate = 2
        self.frames_can_estimate = 7
        self.turning_speed = self.rot_speed
        self.start_time = 0
        self.prev_lines = []
        self.prev_time = None
    
    def set_degree(self, target_deg):
        self.target_rad = np.radians(target_deg, dtype='float32')
    
    def on_start(self, prev_task):
        self.start_time = time.time()

        if self.finish_on_task is not None:
            self.on_start(prev_task)
    
    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        MIN_ROT_RATIO = 0.78

        now_time = time.time()
        elapsed = (now_time - self.start_time) / CLOCK_SLOWDOWN

        if self.finish_on_task is not None:
            self.finish_on_task.on_camera(has_lane, lane, img, lane_mask, green_mask)

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
                    pt1 = int(line[0]), int(line[1])
                    pt2 = int(line[2]), int(line[3])
                    cv2.line(edges, pt1, pt2, (0, 255, 0))

                cv2.imshow('turn', edges)
                cv2.moveWindow('turn', 0, 400)

                
                # convert into vector
                lines = lines[:, :2] - lines[:, 2:]
                lines = nms_vectors(lines, np.radians(5), out_normalized=True)

                # estimate rotating speed
                if 0 < len(self.prev_lines) and 0 < len(lines) and now_time - self.prev_time < 999:
                    dt = (now_time - self.prev_time) / CLOCK_SLOWDOWN

                    similarity = np.dot(lines, self.prev_lines.T)
                    most_similar_curr = np.argmax(similarity)
                    most_similar_prev = most_similar_curr % len(self.prev_lines)
                    most_similar_curr = most_similar_curr // len(self.prev_lines)

                    estimated_speed = np.arccos(similarity[most_similar_curr, most_similar_prev]) / dt

                    # Within seemingly valid range and has correct sign
                    if 0.1 <= np.abs(estimated_speed) <= np.abs(self.rot_speed) * 3 and 0 < estimated_speed * self.rot_speed:
                        # Doesn't work
                        # self.turning_speed = self.turning_speed * 0.8 + estimated_speed * 0.2
                        pass

                # update history
                self.prev_lines = lines
                self.prev_time = now_time
        
        if self.target_rad <= np.abs(self.turning_speed) * elapsed:
            self.completed = True
        
        if self.finish_on_lane and has_lane and self.target_rad * MIN_ROT_RATIO <= np.abs(self.turning_speed) * elapsed:
            print('Completed early by lane')
            self.completed = True
        
        if self.finish_on_task is not None and self.finish_on_task.completed:
            print('Completed early by inner task')
            self.completed = True

        return 0, self.rot_speed


class TurnMovingTask(TaskBase):
    def __init__(self, direction, finish_on_lane=False):
        """
        :param str direction: 회전하는 방향 "left" 혹은 "right"
        :param bool finish_on_lane: 일정 시간 이상 회전한 후 차선이 보이면 완료 처리할지 여부
        """
        super(TurnMovingTask, self).__init__()
        if direction == 'left':
            self.rotate_sign = 1
        else:
            self.rotate_sign = -1
        
        self.finish_on_lane = finish_on_lane
        self.speed = MAX_SPEED / 4
        self.rot_time = 0
        self.rot_speed = 0
        self.start_time = 0
    
    def on_start(self, prev_task):
        dist_to_cross = DIST_CAM_FROM_MASS + 100
        if isinstance(prev_task, WaitCrossTask):
            if prev_task.detected_dist is not None:
                dist_to_cross = prev_task.detected_dist
        
        dist_to_cross *= PIXEL_TO_METER

        # 90 deg * radius
        quarter_circle_arc = 0.5 * np.pi * dist_to_cross
        self.rot_time = quarter_circle_arc / self.speed
        self.rot_speed = self.rotate_sign * 0.5 * np.pi / self.rot_time

        if MAX_STEER < np.abs(self.rot_speed):
            print('WARN: Trying to oversteer (max=%.0f deg, target=%.1f deg)' % (np.degrees(MAX_STEER), np.degrees(self.rot_speed)))

        self.start_time = time.time()

    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        MIN_ROT_RATIO = 0.8

        has_proper_lane = has_lane and np.abs(lane[-2]) < 0.2

        elapsed = (time.time() - self.start_time) / CLOCK_SLOWDOWN
        
        if self.rot_time <= elapsed:
            self.completed = True
            return

        if self.finish_on_lane and has_proper_lane and self.rot_time * MIN_ROT_RATIO <= elapsed:
            print('Completed early by lane')
            # Rotate just one more frame
            self.completed = True
            return self.speed, self.rot_speed

        return self.speed, self.rot_speed


class SleepTask(TaskBase):
    def __init__(self, sec):
        super(SleepTask, self).__init__()
        self.timer = None
        self.sec = sec
    
    def on_camera(self, *args):
        if self.timer is None:
            self.timer = time.time()
        elif self.sec <= (time.time() - self.timer) / CLOCK_SLOWDOWN:
            self.completed = True


class HaltTask(TaskBase):
    def __init__(self, sec=None):
        super(HaltTask, self).__init__()
        self.timer = None
        self.sec = sec
    
    def on_camera(self, *args):
        if self.timer is None:
            self.timer = time.time()
        elif self.sec is not None and self.sec <= (time.time() - self.timer) / CLOCK_SLOWDOWN:
            self.completed = True
        
        return 0, 0


class WaitCrossTask(TaskBase):
    def __init__(self, far=0.9, left=False, right=False):
        """
        :param float far: 0-1 range to detect upto how far away (0=near, 1=far)
        :param bool left: Whether to wait for left side of cross
        :param bool right: Whether to wait for right side of cross
        """
        super(WaitCrossTask, self).__init__()

        if not left and not right:
            raise RuntimeError('Either left or right needs to be present!')
        
        self.near = 1 - far
        self.left = left
        self.right = right
        self.detected_dist = None
    
    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        HEIGHT_WINDOW = 50
        LANE_WIDTH = 100

        if not has_lane:
            return
        
        h, w = lane_mask.shape

        y_far = np.clip(self.near * h - HEIGHT_WINDOW, 0, h - 1).astype('int32')
        y_near = np.clip(self.near * h + HEIGHT_WINDOW, y_far + 1, h).astype('int32')
        roi = lane_mask[y_far:y_near]
        lane_x = np.polyval(lane, np.arange(y_far, y_near, dtype='float32'))

        edges = cv2.Canny(roi, 50, 120)
        visualize = np.tile(np.expand_dims(edges, axis=-1), [1, 1, 3])
        lines = cv2.HoughLinesP(edges, 1, np.radians(0.1), 20, None, 40, 40)

        if lines is None:
            lines = np.empty((0, 4), dtype='float32')
        else:
            lines = lines[:, 0].astype('float32')
        
        direction = lines[:, :2] - lines[:, 2:]
        magnitude = np.linalg.norm(direction, keepdims=True, axis=-1)
        direction /= magnitude
        
        # 해당 선분이 차선의 왼쪽인지 오른쪽인지 표시
        left = np.zeros(len(lines), dtype='bool')
        right = np.zeros(len(lines), dtype='bool')

        lane_left = lane_x - (LANE_WIDTH / 2)
        lane_right = lane_x + (LANE_WIDTH / 2)

        lane_direction = np.asarray([
            np.polyval(lane, y_near) - np.polyval(lane, y_far),
            y_near - y_far
        ], dtype='float32')
        lane_direction /= np.linalg.norm(lane_direction)
        
        for i in range(len(lines)):
            # 차선과 가깝거나 수평인 직선 제거
            if 0.2 <= np.abs(np.dot(direction[i], lane_direction)):
                continue
            
            mean_y = np.mean(lines[i, 1::2])
            lane_idx = np.clip((mean_y - y_far).astype('int32'), 0, len(lane_x))
            if np.all(lane_left[lane_idx] < lines[i, 0::2]) and np.all(lines[i, 0::2] < lane_right[lane_idx]):
                continue
            if np.any(lines[i, 0::2] <= lane_left[lane_idx]):
                left[i] = True
            if np.any(lane_right[lane_idx] <= lines[i, 0::2]):
                right[i] = True

        accepted = None
        
        if self.left and self.right:
            if 2 <= np.sum(left) and 2 <= np.sum(right):
                accepted = left | right
        elif self.left:
            if 2 <= np.sum(left):
                accepted = left
        else:
            if 2 <= np.sum(right):
                accepted = right

        if accepted is not None:
            self.detected_dist = DIST_CAM_FROM_MASS + h - np.max(lines[accepted, 1::2])
            self.completed = True

        # visualize
        for i, line in enumerate(lines):
            pt1 = int(line[0]), int(line[1])
            pt2 = int(line[2]), int(line[3])
            if left[i] and right[i]:
                color = (255, 255, 0)
            elif left[i]:
                color = (255, 0, 0)
            elif right[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.line(visualize, pt1, pt2, color, 2)

        cv2.imshow('cross', visualize)
        cv2.moveWindow('cross', 300, 400)


class WaitGreenMarkerTask(TaskBase):
    def __init__(self, far=None):
        """
        :param float far: 0-1 range to detect upto how far away (0=near, 1=far)
        """
        super(WaitGreenMarkerTask, self).__init__()
        if far is not None:
            self.far = far
        else:
            self.far = 0.5
    
    def on_camera(self, has_lane, lane, img, lane_mask, green_mask):
        KERNEL_SIZE = 35
        LANE_DIST = 100
        THRESHOLD = 180

        if not has_lane:
            return

        img_h = img.shape[0]
        y_start = max(0, int(img_h * (1 - self.far) - KERNEL_SIZE))

        mask = green_mask[y_start:]
        mask = cv2.blur(mask, (KERNEL_SIZE, KERNEL_SIZE))

        mask_idx = np.argmax(mask, axis=-1)

        for y_offset, x in enumerate(mask_idx):
            xlane = np.polyval(lane, y_start + y_offset)
            
            if x < xlane - LANE_DIST or xlane + LANE_DIST < x:
                continue

            if THRESHOLD <= mask[y_offset, x]:
                self.completed = True


class Drive(object):
    def __init__(self):
        self.last_log = time.time()
        self.do_movement = None
        self.lane_detector = SlidingWindowLaneDetector()
        self.lane = None
        self.curr_speed = 0
        self.curr_rot = 0
        self.task_idx = 0
        self.tasks = [
            # TurnStillTask(90, 'left'),
            # HaltTask(),
            # SleepTask(999999),

            WaitCrossTask(left=True, right=True),
            TurnMovingTask('right'),
            WaitGreenMarkerTask(0.2),
            TurnStillTask(90, 'right'),
            HaltTask(0.5),
            TurnStillTask(90, 'right', finish_on_lane=True),

            TurnMovingTask('right'),
            WaitGreenMarkerTask(),
            SleepTask(1),
            # obj 2
            WaitGreenMarkerTask(1.0),
            TurnStillTask(90, 'left'),
            TurnStillTask(90, 'right', finish_on_lane=True),
            WaitGreenMarkerTask(),
            # obj 3
            WaitGreenMarkerTask(),
            # obj 4
            TurnMovingTask('left'),
            WaitGreenMarkerTask(),
            HaltTask(),
        ]

    def on_camera(self, img):
        start_time = time.time()

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
                    self.tasks[self.task_idx].on_start(now_task)
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

        end_time = time.time()
        if 5 <= end_time - self.last_log:
            self.last_log = end_time
            print('Processing took %.1f ms' % (end_time - start_time,))

        if self.do_movement is not None:
            self.do_movement(self.curr_speed, self.curr_rot)


def test_drive():
    from carsim import CarSimulator

    sim = CarSimulator(jitter=False, delay=0)
    sim.reset()

    sim.step(0.5, 0, 1)

    drive = Drive()
    drive.do_movement = sim.step

    while True:
        cam = sim.render()
        drive.on_camera(cam)
        
        if cv2.waitKey(330) & 0xff == ord('q'):
            break
        if cv2.getWindowProperty('img', cv2.WND_PROP_VISIBLE) < 1:
            break

if __name__ == '__main__':
    np.seterr(all='raise')
    test_drive()

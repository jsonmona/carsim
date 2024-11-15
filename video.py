from drive import Drive
from robot import warping

import cv2


def drive_from_video(path):
    def handle_movement(speed, angle):
        pass

    driver = Drive()
    driver.do_movement = handle_movement

    cap = cv2.VideoCapture(path)

    while True:
        _, img = cap.read()

        img = warping(img)

        driver.on_camera(img)

        cv2.waitKey(33)


if __name__ == '__main__':
    drive_from_video('omo2.mp4')
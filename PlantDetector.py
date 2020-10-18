import cv2
import numpy as np
import time
from threading import Thread
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1080))
class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        self.compute()
        out.write(self.frame)
        cv2.imshow('frame', self.frame)
        cv2.waitKey(self.FPS_MS)
    def compute(self):
        lower_blue = np.array([20, 80, 80])
        upper_blue = np.array([50, 255, 255])
        #         # cv2.namedWindow(plant', cv2.WINDOW_AUTOSIZE)
        # image = cv2.GaussianBlur(self.frame, (5, 5), 0, 1)
        image = cv2.blur(self.frame, (3, 3), 0)
        # image = self.frame
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower_blue, upper_blue)
        points = cv2.findNonZero(mask)
        if points is not None:
            rect = cv2.boundingRect(points)
            x, y, w, h = rect
        # result = cv2.bitwise_and(frame, frame, mask=mask)
            _, contours, hierarchy = cv2.findContours(mask,
                                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.rectangle(self.frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.frame, 'Leaf', (int(x-5), int(y-5)),
                        font,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.drawContours(self.frame, contours, -1, (0, 0, 255), 3)
            # cv2.imshow('plant', frame)
            # cv2.waitKey(50)
cap = 'video.mp4'
threaded_camera = ThreadedCamera(cap)
while True:
    try:
        threaded_camera.show_frame()
    except AttributeError:
        pass

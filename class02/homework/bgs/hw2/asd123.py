#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
from iotdemo import motion

# Set OpenCV to use GTK instead of Qt
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import numpy as np

# Force OpenCV to use a headless backend if GUI display is problematic
cv2.setNumThreads(1)

#from openvino.inference_engine import IECore

from iotdemo.motion.motion_detector import MotionDetector

FORCE_STOP = False

def thread_cam1(q):
    # TODO: MotionDetector
    motion1 = MotionDetector()
    motion1.load_preset('motion.cfg')
    cap = cv2.VideoCapture('./resources/conveyor.mp4')
    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.

 ## Load and initialize OpenVINO



    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))
        # TODO: Motion detect
        detected = motion1.detect(frame)
        if detected is None:
            continue
 
        q.put(("VIDEO:Cam1 detected", detected))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    motion2 = MotionDetector()
    motion2.load_preset('motion.cfg')
    cap = cv2.VideoCapture('./resources/conveyor.mp4')
    # TODO: ColorDetector

    # TODO: HW2 Open "resources/conveyor.mp4" video clip

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))
        # TODO: Detect motion 
        detected = motion2.detect(frame)
        if detected is None:
            continue
 
        q.put(("VIDEO:Cam2 detected", detected))
 
    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    try:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if pos:
            cv2.moveWindow(title, pos[0], pos[1])
        cv2.imshow(title, frame)
    except Exception as e:
        print(f"Error displaying {title}: {e}")
        # Save frame to file as fallback
        cv2.imwrite(f"{title.replace(':', '_')}.jpg", frame)


def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py',
    description="Factory tool")

    parser.add_argument("-d",
    "--device",
    default=None,
    type=str,
    help="Arduino port")
    args = parser.parse_args()

    # TODO: HW2 Create a Queue
    q=Queue()
 
    Thread1 = threading.Thread(target=thread_cam1, args=(q,))
    Thread2 = threading.Thread(target=thread_cam2, args=(q,))
    Thread1.start()
    Thread2.start()
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.

    #with FactoryController(args.device) as ctrl:
    while not FORCE_STOP:
        if cv2.waitKey(10) & 0xff == ord('q'):
            break

        # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
        # de-queue name and data
        try:
            name, data = q.get_nowait()
        except Empty:
            continue
        # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
        if name == 'VIDEO:Cam1 live':
            imshow("Cam1 live", data)
        elif name == 'VIDEO:Cam2 live':
            imshow("Cam2 live", data)
 
        elif name == "VIDEO:Cam1 detected":
            imshow("VIDEO:Cam1 Detected", data, pos=(100, 700)) 
    
        elif name == "VIDEO:Cam2 detected":
            imshow("VIDEO:Cam2 Detected", data, pos=(500, 700))



    # TODO: Control actuator, name == 'PUSH'

    if name == 'DONE':
        FORCE_STOP = True

        q.task_done()

    Thread1.join()
    Thread2.join()
 
 
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit(0)
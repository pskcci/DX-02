#!/usr/bin/env python3


import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
from iotdemo import motion

import cv2
import numpy as np
#from openvino.inference_engine import IECore

from iotdemo import MotionDetector
from iotdemo import FactoryController

FORCE_STOP = False

def thread_cam1(q):
    # TODO: MotionDetector
    sMotion = MotionDetector()
    sMotion.load_preset('./resources/motion.cfg')
    
    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture('./resources/conveyor.mp4')
    ## Load and initialize OpenVINO



    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(("VIDEO:Cam1 live", frame))
        # TODO: Motion detect
        detected = sMotion.detect(frame)
        if detected is None:
            continue
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.

        # # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        ## TODO: Inference OpenVINO

        # ## TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        ## TODO: in queue for moving the actuator 1
        q.put(('VIDEO:Cam1 detected', detected))

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    sMotion = MotionDetector()
    sMotion.load_preset('./resources/motion.cfg')
    
    # TODO: ColorDetector

    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture('./resources/conveyor.mp4')
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(("VIDEO:Cam2 live", frame))
        # TODO: Detect motion
        detected = sMotion.detect(frame)
        if detected is None:
            continue
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.

        # # TODO: Detect color

        # # TODO: Compute ratio
        # print(f"{name}: {ratio:.2f}%")

        # # TODO: Enqueue to handle actuator 2
        q.put(('VIDEO:Cam2 detected', detected))
    cap.release()
    q.put(('DONE', None))
    exit()


def imshow(title, frame, pos=None):
    cv2.namedWindow(title)
    if pos:
        cv2.moveWindow(title, pos[0], pos[1])
    cv2.imshow(title, frame)


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
            name, data = q.get_nowait()  # 큐에서 데이터 가져오기
        except Empty:
            continue

        if name == 'VIDEO:Cam1 live':
            imshow("cam1 live", data)

        elif name == 'VIDEO:Cam2 live':
            imshow("cam2 live", data)
            
        elif name == 'VIDEO:Cam1 detected':
        
            print(f"Detected: {data}")  # 감지된 데이터를 출력
            imshow("VIDEO:Cam1 detected", data)  # 감지된 결과를 화면에 출력
        elif name == 'VIDEO:Cam2 detected':
            imshow("VIDEO:Cam2 detected", data)  # 감지된 결과를 화면에 출력
            
            
            
        # TODO: Control actuator, name == 'PUSH'
        
        elif name == 'DONE':
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
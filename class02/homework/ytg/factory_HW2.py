#!/usr/bin/env python3

import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep
from threading import Thread

import cv2
import numpy as np
#from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False


def thread_cam1(q):
    # MotionDetector 초기화
    sMotion = MotionDetector()
    #sMotion.load.preset("/home/test/workspace/smart-factory/motion.cfg")
    sMotion.load_preset("/home/test/workspace/smart-factory/motion.cfg")

    # 동영상 파일 열기
    cap = cv2.VideoCapture("/home/test/workspace/smart-factory/conveyor.mp4")  # 동영상 파일 경로 지정

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame))

        # TODO: Motion detect
        detect_frame = sMotion.detect(frame)  # 전경 마스크 생성

        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detect_frame))

        # TODO: Inference OpenVINO

        # TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()



def thread_cam2(q):
    # MotionDetector 초기화
    sMotion = MotionDetector()
    sMotion.load_preset("/home/test/workspace/smart-factory/motion.cfg")

    # ColorDetector 초기화

    # 동영상 파일 열기
    cap = cv2.VideoCapture("/home/test/workspace/smart-factory/conveyor.mp4")  # 동영상 파일 경로 지정

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break

        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame))

        # TODO: Detect motion
        detect_frame = sMotion.detect(frame)  # 전경 마스크 생성

        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detect_frame))

        # TODO: Detect color

        # TODO: Compute ratio
        # print(f"{name}: {ratio:.2f}%") 

        # TODO: Enqueue to handle actuator 2

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

    # 큐 생성
    q = Queue()

    # 스레드 생성 및 시작
    t1 = Thread(target=thread_cam1, args=(q,))
    t2 = Thread(target=thread_cam2, args=(q,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                FORCE_STOP = True
                break

            try:
                name, data = q.get_nowait()
                if data is not None and data.size > 0: # 프레임 유효한지 확인
                    if name[-4:] == 'live':
                        imshow(name[6:], data)
                    elif name[-8:] == 'detected':
                        imshow(name[6:], data)  # 모션 감지 결과 표시
                elif name == 'DONE':
                    FORCE_STOP = True
                q.task_done()
            except Empty:
                pass

    t1.join()
    t2.join()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    #try:
    # except Exception:
    #     os._exit()


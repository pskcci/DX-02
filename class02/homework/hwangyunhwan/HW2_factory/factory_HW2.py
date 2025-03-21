#!/usr/bin/env python3
import os
import threading
import queue
from argparse import ArgumentParser
from time import sleep

import cv2
import numpy as np

from iotdemo import FactoryController, MotionDetector

FORCE_STOP = False

def thread_cam1(q):
    cMotion = MotionDetector()
    cMotion.load_preset("/home/yun/workspace/smart-factory/resources/motion.cfg")
    cap = cv2.VideoCapture("/home/yun/workspace/smart-factory/resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret:
            break

        # detect() 함수가 None을 반환할 수 있으므로 이를 체크
        motion_detected = cMotion.detect(frame)
        if motion_detected is not None and motion_detected.any():  # 수정된 부분
            q.put(("Cam1_detected", frame))  # 움직임 감지 시 저장

        q.put(("cam1", frame))  # Cam2 영상 큐 저장

    cap.release()
    q.put(('DONE', None))  # 종료 신호 전달

#         if cMotion.detect(frame):
#             q.put(("Cam1_detected", frame))  # 움직임 감지 시 저장

#         q.put(("cam1", frame))  # Cam1 영상 큐 저장


def thread_cam2(q):
    cMotion = MotionDetector()
    cMotion.load_preset("/home/yun/workspace/smart-factory/resources/motion.cfg")
    cap = cv2.VideoCapture("/home/yun/workspace/smart-factory/resources/conveyor.mp4")

    while not FORCE_STOP:
        sleep(0.03)
        ret, frame = cap.read()
        if not ret:
            break

        # detect() 함수가 None을 반환할 수 있으므로 이를 체크
        motion_detected = cMotion.detect(frame)
        if motion_detected is not None and motion_detected.any():  # 수정된 부분
            q.put(("Cam2_detected", frame))  # 움직임 감지 시 저장

        q.put(("cam2", frame))  # Cam2 영상 큐 저장

    cap.release()
    q.put(('DONE', None))  # 종료 신호 전달

def main():
    global FORCE_STOP

    parser = ArgumentParser(prog='python3 factory.py', description="Factory tool")
    parser.add_argument("-d", "--device", default=None, type=str, help="Arduino port")
    args = parser.parse_args()

    shared_queue = queue.Queue()

    # 스레드 시작
    t1 = threading.Thread(target=thread_cam1, args=(shared_queue,))
    t2 = threading.Thread(target=thread_cam2, args=(shared_queue,))
    t1.start()
    t2.start()

    with FactoryController(args.device) as ctrl:
        done_count = 0  # 'DONE' 신호 카운트 (2개 오면 종료)
        
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                FORCE_STOP = True
                break

            try:
                # 큐에서 데이터 가져오기
                name, frame = shared_queue.get(timeout=1)
            except queue.Empty:
                continue  # 큐가 비어 있으면 건너뛰기

            if name == 'DONE':  # 종료 신호 감지
                done_count += 1
                if done_count == 2:  # 두 개의 스레드가 모두 종료될 때까지 대기
                    FORCE_STOP = True
                continue

            # Cam1과 Cam2의 영상을 개별 창에 표시
            if name == "cam1":
                cv2.imshow("Cam1", frame)
            elif name == "cam2":
                cv2.imshow("Cam2", frame)
            elif name == "Cam1_detected":  
                cv2.imshow("Cam1 Detected", frame)
            elif name == "Cam2_detected":  # 수정된 부분
                cv2.imshow("Cam2 Detected", frame)

            shared_queue.task_done()  # task_done() 올바른 위치

    # 스레드 종료 대기
    t1.join()
    t2.join()

    cv2.destroyAllWindows()  # 모든 창 닫기

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)  # 오류 발생 시 강제 종료

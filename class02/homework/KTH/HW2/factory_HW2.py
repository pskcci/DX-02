import os
import threading
from argparse import ArgumentParser
from queue import Empty, Queue
from time import sleep

import cv2
import numpy as np
# from openvino.inference_engine import IECore

from iotdemo import FactoryController, MotionDetector # iotdemo라는 파일로부터 각 클래스 FactoryController, MotionDetector를 불러온다.

FORCE_STOP = False


def thread_cam1(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    # TODO: Load and initialize OpenVINO
    cMotion.load_preset("/home/potato/workspace/smart-factory/resources/motion.cfg")
    # TODO: HW2 Open video clip resources/conveyor.mp4 instead of camera device.
    cap = cv2.VideoCapture("/home/potato/workspace/smart-factory/resources/conveyor.mp4") # 감자추가 절대경로방식
    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # TODO: HW2 Enqueue "VIDEO:Cam1 live", frame info
        q.put(('VIDEO:Cam1 live', frame)) #지피티 개선안 적용
        # TODO: Motion detect
        detected = cMotion.detect(frame)
        if detected is None:
            continue
        # TODO: Enqueue "VIDEO:Cam1 detected", detected info.
        q.put(('VIDEO:Cam1 detected', detected))
        # # abnormal detect
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # reshaped = detected[:, :, [2, 1, 0]]
        # np_data = np.moveaxis(reshaped, -1, 0)
        # preprocessed_numpy = [((np_data / 255.0) - 0.5) * 2]
        # batch_tensor = np.stack(preprocessed_numpy, axis=0)

        # # TODO: Inference OpenVINO

        # # TODO: Calculate ratios
        # print(f"X = {x_ratio:.2f}%, Circle = {circle_ratio:.2f}%")

        # # TODO: in queue for moving the actuator 1

    cap.release()
    q.put(('DONE', None))
    exit()


def thread_cam2(q):
    # TODO: MotionDetector
    cMotion = MotionDetector()
    # TODO: ColorDetector
    cMotion.load_preset("/home/potato/workspace/smart-factory/resources/motion.cfg")
    # TODO: HW2 Open "resources/conveyor.mp4" video clip
    cap = cv2.VideoCapture("/home/potato/workspace/smart-factory/resources/conveyor.mp4") # 감자추가 절대경로방식

    while not FORCE_STOP:
        sleep(0.03)
        _, frame = cap.read()
        if frame is None:
            break
        # TODO: HW2 Enqueue "VIDEO:Cam2 live", frame info
        q.put(('VIDEO:Cam2 live', frame)) #지피티 개선한 적용
        # TODO: Detect motion
        detected = cMotion.detect(frame)
        if detected is None:
            continue
        # TODO: Enqueue "VIDEO:Cam2 detected", detected info.
        q.put(('VIDEO:Cam2 detected', detected))
        # TODO: Detect color

        # TODO: Compute ratio
        # print(f"{name}: {ratio:.2f}%") #단순 재생만 필요하니 일단 각주화

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
    
    # TODO: HW2 Create a Queue
    q = Queue() #교수님 hint 적용
    # TODO: HW2 Create thread_cam1 and thread_cam2 threads and start them.
    t1 = threading.Thread(target=thread_cam1, args=(q,)) #감자추가+지피티 개선안 적용+교수님 hint 적용
    t2 = threading.Thread(target=thread_cam2, args=(q,)) #감자추가+지피티 개선안 적용+교수님 hint 적용

    t1.start() #감자추가
    t2.start() #감자추가

    with FactoryController(args.device) as ctrl:
        while not FORCE_STOP:
            if cv2.waitKey(10) & 0xff == ord('q'):
                break

            # TODO: HW2 get an item from the queue. You might need to properly handle exceptions.
            # de-queue name and data
            try:
                name, data = q.get_nowait()
                # Dictionary to map names to functions
                # Call the appropriate function based on the name
            except Empty:
                continue
            # TODO: HW2 show videos with titles of 'Cam1 live' and 'Cam2 live' respectively.
            if name [-4:]== 'live':
                imshow(name[6:], data)
            elif name [11:]=='detected':
                imshow(name[6:], data)
            # TODO: Control actuator, name == 'PUSH'
            elif name == 'DONE':
                FORCE_STOP = True
        q.task_done()
    
    cv2.destroyAllWindows()
    
    t1.join() #감자추가+지피티 개선안 적용
    t2.join() #감자추가+지피티 개선안 적용


if __name__ == '__main__':
    try:
        main()
    except Exception:
        os._exit()

from iotdemo import FactoryController
import time


with FactoryController('/dev/ttyACM0') as ctrl:
    ctrl.push_actuator(1)
    print("actuator1 on")
    time.sleep(1.5)

    ctrl.red = False
    print("RED on")
    time.sleep(1.5)
    ctrl.red = True
    print("RED off")
    time.sleep(1.5)

    ctrl.orange = False
    print("orange on")
    time.sleep(1.5)
    ctrl.orange = True
    print("orange off")
    time.sleep(1.5)

    ctrl.green = False
    print("green on")
    time.sleep(1.5)
    ctrl.green = True
    print("green off")
    time.sleep(1.5)
    
ctrl.close()

    # BEACON_RED = 2
    # BEACON_ORANGE = 3
    # BEACON_GREEN = 4

    # BEACON_BUZZER = 5

    # ACTUATOR_1 = 6
    # ACTUATOR_2 = 7

    # CONVEYOR_EN = 8
    # CONVEYOR_PWM = 9
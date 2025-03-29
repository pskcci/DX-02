import time
from iotdemo import FactoryController

with FactoryController('/dev/ttyACM0') as ctrl:
    ## 코드 작성
    ctrl.red = False
    ctrl.green = True
    ctrl.orange = True
    ctrl.push_actuator(1)
    #time.sleep(3)

ctrl.close()



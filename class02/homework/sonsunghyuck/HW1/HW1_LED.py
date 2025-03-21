
from iotdemo import FactoryController
import time

with FactoryController('/dev/ttyACM0') as ctrl:
    ctrl.red = False
    time.sleep(0.1)
    ctrl.green = False
    time.sleep(0.1)
    ctrl.orange = False
    time.sleep(0.1)

    ctrl.push_actuator(1)
    ctrl.push_actuator(2)
    time.sleep(2)
    
ctrl.close()

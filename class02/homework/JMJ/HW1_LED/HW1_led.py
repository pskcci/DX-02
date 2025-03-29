from iotdemo import FactoryController
import time

count=0

with FactoryController('/dev/ttyACM0') as ctrl:
    ## 코드 작성
    while True:
        count += 1
        print(count)
        ctrl.red=True
        ctrl.green=True
        time.sleep(1)
        ctrl.red=False
        ctrl.green=False
        time.sleep(1)
             
        if count == 5:            
            print('Finish')
            break

ctrl.close()
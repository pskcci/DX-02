from iotdemo import FactoryController
from time import sleep
red = 0
orange = 0
green = 0
with FactoryController('/dev/ttACM0') as ctrl:
     
    #코드 작성
    while True:
        # ctrl.red = True
        # ctrl.orange = True
        # ctrl.green = True
        idx = int(input())
        if idx == 3:
            red += 1
            red = red % 2
            print(bool(red))
            ctrl.red = bool(red)
        if idx == 4:
            orange += 1
            orange = orange % 2
            print(bool(orange))
            ctrl.orange = bool(orange)            
        if idx == 5:
            green += 1
            green = green % 2
            print(bool(green))
            ctrl.green = bool(green)                        
        if idx == 99:
            break

ctrl.close()
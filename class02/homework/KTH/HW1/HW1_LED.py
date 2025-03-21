from iotdemo import FactoryController
import time

with FactoryController('/dev/ttyACM0') as ctrl:
    #코드 작성
    # 아두이노 특성에 따라 LEF는 False에 켜지고 True에 꺼진다.
    ctrl.red = False     
    time.sleep(3)
    ctrl.orange = False 
    time.sleep(3)
    ctrl.orange = True  
    time.sleep(3)
    ctrl.orange = False  
    time.sleep(3)
    ctrl.green = False   
    time.sleep(3)
    
    # ctrl.push_actuator(1) 
    # ctrl.push_actuator(2)
    
    print("정상 실행완료")

ctrl.close()
print("종료완료 \n 종료이후 디폴트 상태\n RED [off] │ orange [on] │ green [on] │ artuator 1, 2 [on] │ artuator 2 [on]") 

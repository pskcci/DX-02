from iotdemo import FactoryController
import time

# FactoryController 객체 생성
ctrl = FactoryController('/dev/ttyACM0')

# LED 상태 설정
ctrl.red = False
ctrl.orange = False
ctrl.green = True
ctrl.push_actuator(1)

# 프로그램이 종료되지 않도록 계속 대기
try:
    while True:
        time.sleep(1)  # 1초 대기 (계속 실행되도록 대기)
except KeyboardInterrupt:
    # 사용자 인터럽트(CTRL+C)로 종료 시 처리
    print("프로그램 종료")
finally:
    # 연결 종료
    ctrl.close()

import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from roboticstoolbox import ET
import numpy as np
import time

# 1. ETS 정의 (사용자 모델)
e = ET.Ry(jindex=0) * ET.ty(0.05)
e *= ET.Rx(jindex=1) * ET.tz(-0.05)
e *= ET.Rz(jindex=2) * ET.tz(-0.07)
e *= ET.Ry(jindex=3) * ET.tx(0.05)
e *= ET.Ry(jindex=4) * ET.tx(0.05)
e *= ET.Rx(jindex=5) * ET.tx(0.02)

my_robot = rtb.ERobot(e, name="My_Design")

# 2. 시각화 초기화 (문제가 되는 backend 명시와 teach를 아예 제거)
# block=False로 설정하여 아래의 while 루프가 즉시 실행되게 합니다.
env = my_robot.plot([0] * 6, jointaxes=True, block=False)

print("🚀 시뮬레이션 시작! 루프가 돌며 로봇이 움직이는지 확인하세요.")

try:
    current_q = np.zeros(6)
    loop_count = 0

    while True:
        loop_count += 1
        # 새로운 랜덤 목표 각도
        q_target = (np.random.rand(6) - 0.5) * np.pi

        steps = 10  # 빠른 확인을 위해 스텝 축소
        for i in range(steps):
            q_now = current_q + (q_target - current_q) * (i / steps)

            # 3. 화면 강제 업데이트 (가장 안전한 방식)
            env.q = q_now

            # --- 엔드이펙터 포즈(위치 + 자세) 계산 ---
            T = my_robot.fkine(q_now)
            pos = T.t  # 위치 (x, y, z)
            rpy = T.rpy(unit='deg')  # 자세 (Roll, Pitch, Yaw)

            # 터미널 출력: 루프 카운트 + 위치 + 자세
            print(
                f"[{loop_count:03d}] 📍 X:{pos[0]:.2f} Y:{pos[1]:.2f} Z:{pos[2]:.2f} | 🔄 R:{rpy[0]:.1f}° P:{rpy[1]:.1f}° Y:{rpy[2]:.1f}° ",
                end='\r')

            # GUI 엔진에게 그릴 시간을 줌
            plt.pause(0.001)

        current_q = q_target
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n👋 종료합니다.")
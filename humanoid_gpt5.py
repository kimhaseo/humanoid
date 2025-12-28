import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
import matplotlib.pyplot as plt
import time

# 1. 로봇 모델 정의
links = [
    # 1. Shoulder Pitch: alpha=0으로 둡니다. (이미 base에서 눕힐 것이기 때문)
    rtb.RevoluteDH(d=0, a=0, alpha=np.pi / 2, ),
    rtb.RevoluteDH(d=0.05, a=0, alpha=np.pi / 2, offset=0),
    # rtb.RevoluteDH(d=0.1, a=0.1, alpha=-np.pi / 2),
    # rtb.RevoluteDH(d=0, a=0.1, alpha=np.pi / 2),
    # rtb.RevoluteDH(d=0, a=0, alpha=np.pi / 2),
    # rtb.RevoluteDH(d=0.1, a=0, alpha=0)
]

robot = rtb.DHRobot(links, name='Humanoid_Arm')

# [핵심 수정]
# 1. 위치를 0.2m 올리고 (Trans)
# 2. X축 기준으로 90도 회전시켜서(Rx) 1번 관절 축을 옆으로 눕힙니다.
# 이렇게 해야 1번 관절이 베이스 Z축(하늘)이 아닌 옆을 축으로 '앞뒤'로 돕니다.
robot.base = SE3.Trans(0, 0, 0.2) * SE3.Rx(np.pi / 2)

# 2. 시각화 및 루프 (기존과 동일)
q_current = np.zeros(6)
env = robot.plot(q_current, backend='pyplot', jointaxes=True, block=False)

print("🎬 진짜 어깨 앞뒤 회전 테스트 시작...")

t = 0
try:
    while True:
        x = 0.15 + 0.05 * np.cos(t)
        y = 0.05 + 0.05 * np.sin(t)
        z = 0.15 + 0.03 * np.sin(2 * t)

        T_target = SE3.Trans(x, y, z) * SE3.RPY(0, np.radians(45), 0)
        sol = robot.ikine_LM(T_target, q0=q_current, mask=[1, 1, 1, 1, 1, 0])

        if sol.success:
            q_current = sol.q
            robot.q = q_current
            env.step(0.001)

        t += 0.04
        time.sleep(0.01)
        if not plt.fignum_exists(plt.gcf().number): break
except KeyboardInterrupt:
    pass
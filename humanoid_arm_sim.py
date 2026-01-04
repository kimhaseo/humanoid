import roboticstoolbox as rtb
from roboticstoolbox import ET
from spatialmath import SE3
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# 1. ETS 정의 (qlim 파라미터 추가)
# 각 관절의 한계를 [최소 라디안, 최대 라디안]으로 설정합니다.
# 예: np.radians(-150), np.radians(150)
lim = [-np.pi * 150 / 180, np.pi * 150 / 180]

e = ET.Ry(jindex=0, qlim=lim) * ET.ty(0.05)
e *= ET.Rx(jindex=1, qlim=lim) * ET.tz(-0.05)
e *= ET.Rz(jindex=2, qlim=lim) * ET.tz(-0.1)
e *= ET.Ry(jindex=3, qlim=lim) * ET.tx(0.1)
e *= ET.Ry(jindex=4, qlim=lim) * ET.tx(0.03)
e *= ET.Rx(jindex=5, qlim=lim) * ET.tx(0.01)

my_robot = rtb.ERobot(e, name="My_Limited_Robot")
q_current = np.zeros(6)

# 2. 시각화 설정
env = my_robot.plot(q_current, backend='pyplot', jointaxes=True, block=False)

print("🎬 관절 제한(qlim) 적용 시뮬레이션 시작...")

t = 0
try:
    while True:
        # 3. 목표 궤적 계산
        target_x = 0.05 + 0.03 * np.cos(t)
        target_y = 0.05 + 0.03 * np.sin(t)
        target_z = -0.05

        # 4. 자세 고정 (Z축 상방)
        T_target = SE3.Trans(target_x, target_y, target_z) * SE3.RPY(0, np.radians(-90), 0)

        # 5. 역운동학(IK) 수행
        # ikine_LM은 모델에 정의된 qlim을 자동으로 인식하여 최적해를 찾습니다.
        sol = my_robot.ikine_LM(T_target, q0=q_current)

        if sol.success:
            q_current = sol.q
            my_robot.q = q_current
            env.step(0.001)

            q_deg = np.degrees(q_current)

            # 실시간 출력
            msg = f"\r⚙️ Q(deg): Q0:{q_deg[0]:5.1f}, Q1:{q_deg[1]:5.1f}, Q2:{q_deg[2]:5.1f}, Q3:{q_deg[3]:5.1f}, Q4:{q_deg[4]:5.1f}, Q5:{q_deg[5]:5.1f}"
            sys.stdout.write(msg)
            sys.stdout.flush()
        else:
            # IK가 실패한 경우 (가동 범위를 벗어났거나 특이점인 경우)
            sys.stdout.write("\r⚠️ Warning: Target out of reach or joint limit!          ")
            sys.stdout.flush()

        t += 0.04
        time.sleep(0.01)

        if not plt.fignum_exists(plt.gcf().number):
            break

except KeyboardInterrupt:
    sys.stdout.write("\n\n👋 시뮬레이션을 종료합니다.\n")
    sys.stdout.flush()
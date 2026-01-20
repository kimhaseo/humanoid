import roboticstoolbox as rtb
from roboticstoolbox import ET
from spatialmath import SE3
import numpy as np
import matplotlib.pyplot as plt
import sys

# 1. 로봇 정의 (기존과 동일)
lim = [-np.pi * 150 / 180, np.pi * 150 / 180]
e = ET.Ry(jindex=0, qlim=lim) * ET.ty(0.05)
e *= ET.Rx(jindex=1, qlim=lim) * ET.tz(-0.05)
e *= ET.Rz(jindex=2, qlim=lim) * ET.tz(-0.1)
e *= ET.Ry(jindex=3, qlim=lim) * ET.tx(0.1)
e *= ET.Ry(jindex=4, qlim=lim) * ET.tx(0.03)
e *= ET.Rx(jindex=5, qlim=lim) * ET.tx(0.01)

my_robot = rtb.ERobot(e, name="My_Smooth_Robot")
q_current = np.zeros(6)

# 2. 시각화 설정
env = my_robot.plot(q_current, backend='pyplot', jointaxes=True, block=False)

print("🚀 보간(Interpolation) 적용 시뮬레이션 시작...")

# 보간 설정
steps = 100  # 현재 위치에서 다음 목표까지의 분할 단계 (클수록 부드러움)

t_cycle = 0
try:
    while True:
        # 3. 목표 지점 계산 (원형 궤적)
        target_x = 0.05 + 0.03 * np.cos(t_cycle)
        target_y = 0.05 + 0.03 * np.sin(t_cycle)
        target_z = -0.15
        T_target = SE3.Trans(target_x, target_y, target_z) * SE3.RPY(0, np.radians(90), 0)

        # 4. 역운동학(IK) 수행 - 최종 목표 각도(q_goal) 찾기
        sol = my_robot.ikine_LM(T_target, q0=q_current)

        if sol.success:
            q_goal = sol.q

            # 5. JTRAJ를 이용한 보간 실행
            # q_current에서 q_goal까지 'steps'만큼 부드러운 경로 생성
            traj = rtb.jtraj(q_current, q_goal, steps)

            # 6. 생성된 궤적을 따라 미세 이동
            for q_step in traj.q:
                q_current = q_step
                my_robot.q = q_current
                env.step(0.01)  # 시뮬레이션 갱신

                # 실시간 출력 (deg)
                q_deg = np.degrees(q_current)
                msg = f"\r⚙️ Smooth Moving: Q0:{q_deg[0]:5.1f}, Q1:{q_deg[1]:5.1f}, Q2:{q_deg[2]:5.1f}"
                sys.stdout.write(msg)
                sys.stdout.flush()

        else:
            sys.stdout.write("\r⚠️ Warning: Out of reach!                          ")
            sys.stdout.flush()

        t_cycle += 1  # 궤적 진행 속도
        if not plt.fignum_exists(plt.gcf().number):
            break

except KeyboardInterrupt:
    print("\n👋 시뮬레이션 종료")
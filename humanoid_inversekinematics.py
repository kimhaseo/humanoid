import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3

links = [
    rtb.RevoluteDH(d=0.05, a=0, alpha=np.pi / 2 ,offset=np.pi / 2),
    rtb.RevoluteDH(d=0, a=-0.05, alpha=0, offset=0),
    rtb.RevoluteDH(d=0.05, a=0, alpha=0)
]

robot = rtb.DHRobot(links, name='Shoulder_3DOF')

# 베이스를 눕혀서 1번 관절이 옆을 보게 시작
robot.base = SE3.Rx(np.pi / 2)

print("✅ 3번 관절(Yaw) 추가 완료")
print("- 1번(Pitch): 옆을 봄")
print("- 2번(Roll): 앞을 봄")
print("- 3번(Yaw): 아래(팔 방향)를 봄")

# 차렷 자세 [q1, q2, q3] = [0, 0, 0] 시각화
robot.plot([0, 0, 0], backend='pyplot', jointaxes=True, block=True)
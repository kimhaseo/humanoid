# 필요한 라이브러리: pip install roboticstoolbox-python numpy

import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3

## 1. D-H 파라미터 정의 및 로봇 모델 생성
# 이 파라미터는 현재 사용자의 코드가 성공한 '예시' 값입니다.
# 실제 로봇 설계 시 아래 값을 정확하게 측정하여 수정해야 합니다.
# DH 파라미터: [alpha, a, theta_offset, d]
links = [
    # L1 (어깨 베이스): d=0.2 (베이스 높이), alpha=pi/2
    rtb.RevoluteDH(d=0.2, a=0, alpha=np.pi / 2, offset=0),
    # L2 (상완 링크): a=0.3 (상완 길이)
    rtb.RevoluteDH(d=0, a=0.3, alpha=0, offset=0),
    # L3 (팔꿈치 오프셋 전 링크): a=0.2
    rtb.RevoluteDH(d=0, a=0.2, alpha=np.pi / 2, offset=0),
    # L4 (팔뚝 링크): d=0.2
    rtb.RevoluteDH(d=0.2, a=0, alpha=-np.pi / 2, offset=0),
    # L5 (손목 피치)
    rtb.RevoluteDH(d=0, a=0, alpha=np.pi / 2, offset=0),
    # L6 (손목 롤): d=0.1 (엔드 이펙터까지의 최종 오프셋)
    rtb.RevoluteDH(d=0.1, a=0, alpha=0, offset=0)
]
# 6개의 링크로 구성된 로봇 모델을 생성합니다.
robot = rtb.DHRobot(links, name='Custom_6DOF_Arm')


def solve_inverse_kinematics(target_position: list, target_rpy: list, initial_guess=None):
    """
    목표 위치(XYZ)와 자세(RPY)에 대한 역기구학을 계산하는 함수입니다.

    :param target_position: 목표 [x, y, z] 리스트 또는 배열 (m)
    :param target_rpy: 목표 [Roll, Pitch, Yaw] 리스트 또는 배열 (deg)
    :param initial_guess: IK 솔버가 시작할 초기 관절 각도 [q1, q2, ..., q6] (rad)
    :return: 6개의 관절 각도 (rad) 또는 None (실패 시)
    """

    # 목표 자세 (Roll, Pitch, Yaw)를 도(deg)에서 라디안(rad)으로 변환
    target_rpy_rad = np.radians(target_rpy)

    # 1. 목표 포즈 (Target Pose) 정의 (4x4 동차 변환 행렬 T_target)
    # SE3.Trans()로 위치, SE3.RPY() 생성자로 자세를 정의하여 행렬을 생성
    T_target = SE3.Trans(target_position[0], target_position[1], target_position[2]) * \
               SE3.RPY(target_rpy_rad, unit='rad')

    # 초기 관절 각도 설정 (시작점)
    if initial_guess is None:
        q0 = np.array([0, 0, 0, 0, 0, 0])
    else:
        q0 = np.array(initial_guess)

    print(f"--- IK 계산 시작 ---")
    print(f"목표 위치 (XYZ): {target_position} m")
    print(f"목표 자세 (RPY): {target_rpy} deg")

    # 2. 역기구학(IK) 계산 실행 (Levenberg-Marquardt 알고리즘 사용)
    # 이 수치적 솔버가 자코비안 행렬을 반복적으로 사용하여 해를 찾습니다.
    sol = robot.ikine_LM(
        T_target,
        q0=q0,
        ilimit=500,  # 최대 반복 횟수
        tol=1e-6,  # 오차 허용 한계
        mask=[1, 1, 1, 1, 1, 1]  # 6자유도 모두(위치 3개, 자세 3개) 고려
    )

    # 3. 결과 반환 및 검증
    if sol.success:
        q_solution = sol.q  # 최종 계산된 6개 관절 각도 (라디안)

        # 검증을 위한 정기구학(FK) 수행
        T_achieved = robot.fkine(q_solution)
        position_error = np.linalg.norm(T_target.t - T_achieved.t)

        print("\n✅ IK 계산 성공")
        print(f"최종 관절 각도 (라디안): {q_solution}")
        print(f"최종 관절 각도 (도): {np.degrees(q_solution)}")
        print(f"도달 위치 오차 (Norm): {position_error:.6f} m")

        return q_solution
    else:
        print("\n❌ IK 해를 찾는 데 실패했습니다. 목표 포즈가 작업 공간을 벗어났거나 특이점 근처일 수 있습니다.")
        return None


# ... (중략: solve_inverse_kinematics 함수 정의)

# ==========================================================
# 🚀 함수 실행 예시 (테스트 버전)
# ==========================================================

# 1. 목표 위치 [X, Y, Z] (미터)

# **로봇 베이스에 더 가깝게 목표 설정**
TARGET_POS_TEST = [0.2, 0.1, 0.25]

# 2. 목표 자세 [Roll, Pitch, Yaw] (도)
TARGET_RPY_TEST = [50, 0, 0] # 단순한 자세로 설정

# 3. 초기 추측값 (Optional, 특이점 회피 시도)# ... (중략: solve_inverse_kinematics 함수 정의)
# # ==========================================================
# # 🚀 함수 실행 예시 (테스트 버전)
# # ==========================================================
#
# # 1. 목표 위치 [X, Y, Z] (미터)
# # **로봇 베이스에 더 가깝게 목표 설정**
# TARGET_POS_TEST = [0.2, 0.1, 0.3]
#
# # 2. 목표 자세 [Roll, Pitch, Yaw] (도)
# TARGET_RPY_TEST = [0, 0, 0] # 단순한 자세로 설정
#
# # 3. 초기 추측값 (Optional, 특이점 회피 시도)
# INITIAL_Q_GUESS = [0.1, 0.1, 0.1, 0, 0, 0]
#
# # IK 계산 실행
# print("\n========== 테스트 1: 목표 위치 단순화 ==========")
# solution_q = solve_inverse_kinematics(TARGET_POS_TEST, TARGET_RPY_TEST, INITIAL_Q_GUESS)
INITIAL_Q_GUESS = [0.1, 0.1, 0.1, 0, 0, 0]

# IK 계산 실행
print("\n========== 테스트 1: 목표 위치 단순화 ==========")
solution_q = solve_inverse_kinematics(TARGET_POS_TEST, TARGET_RPY_TEST, INITIAL_Q_GUESS)
import numpy as np
from scipy.optimize import minimize

# --- 회전 행렬 함수들 ---
# 로봇 각 관절의 회전축별로 회전 행렬을 만들어주는 함수들입니다.
# 각도 theta(라디안)를 입력하면 3x3 회전 행렬을 반환합니다.

def rotation_y(theta):
    # y축을 중심으로 theta만큼 회전하는 행렬
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],   # x축 방향이 cos, sin으로 변함
        [0, 1, 0],   # y축은 변하지 않음
        [-s, 0, c]   # z축 방향도 cos, sin으로 변함
    ])

def rotation_x(theta):
    # x축을 중심으로 theta만큼 회전하는 행렬
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],    # x축은 변하지 않음
        [0, c, -s],   # y축과 z축이 cos, sin으로 회전함
        [0, s, c]
    ])

def rotation_z(theta):
    # z축을 중심으로 theta만큼 회전하는 행렬
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],   # x축과 y축이 cos, sin으로 회전함
        [s, c, 0],
        [0, 0, 1]     # z축은 변하지 않음
    ])

# --- 순방향 운동학 함수 (Forward Kinematics, FK) ---
# 5자유도 어깨 관절 구조에 대해 각 관절 각도를 넣으면 엔드포인트 위치와
# 자세(회전 행렬)를 계산해주는 함수입니다.

def fk_5dof_shoulder_custom(q1_deg, q2_deg, q3_deg, q4_deg, q5_deg):
    # 입력 각도는 degree 단위라서 radian으로 변환
    q1 = np.radians(q1_deg)  # 관절1: y축 회전 (앞뒤 회전)
    q2 = np.radians(q2_deg)  # 관절2: x축 회전 (좌우 들기)
    q3 = np.radians(q3_deg)  # 관절3: z축 회전 (비틀림)
    q4 = np.radians(q4_deg)  # 관절4: y축 회전
    q5 = np.radians(q5_deg)  # 관절5: y축 회전

    # 각 관절 회전 행렬 생성
    R1 = rotation_y(q1)
    R2 = rotation_x(q2)
    R3 = rotation_z(q3)
    R4 = rotation_y(q4)
    R5 = rotation_y(q5)

    # 관절2 위치는 관절1 기준으로 y축 방향으로 50mm (0.05m) 떨어져 있음
    offset_joint2 = np.array([0, 0.05, 0])
    # 관절1 회전 행렬을 적용하여 관절2의 월드 좌표 위치 계산
    pos_joint2 = R1 @ offset_joint2

    # 관절2 끝단 링크 벡터: z축 아래 방향으로 60mm (0,0,-0.06)m
    link2_vector = np.array([0, 0, -0.06])
    # 관절2 회전 행렬과 관절1 회전 행렬을 곱해 링크 벡터에 회전 적용 후 위치 계산
    pos_joint3 = pos_joint2 + R1 @ R2 @ link2_vector

    # 관절3 끝단 링크 벡터도 z축 아래 60mm
    link3_vector = np.array([0, 0, -0.06])
    # 관절3 회전 행렬과 이전 행렬들을 곱해 링크 벡터에 회전 적용 후 위치 계산
    pos_joint4 = pos_joint3 + R1 @ R2 @ R3 @ link3_vector

    # 관절4 끝단 링크 벡터는 x축 방향으로 70mm
    link4_vector = np.array([0.07, 0, 0])
    pos_joint5 = pos_joint4 + R1 @ R2 @ R3 @ R4 @ link4_vector

    # 관절5 끝단 링크 벡터는 x축 방향으로 50mm
    link5_vector = np.array([0.05, 0, 0])
    # 마지막 회전 행렬까지 모두 곱해 링크 벡터에 회전 적용 후 최종 엔드포인트 위치 계산
    end_pos = pos_joint5 + R1 @ R2 @ R3 @ R4 @ R5 @ link5_vector

    # 최종 엔드포인트의 회전 행렬 (자세)
    R_end = R1 @ R2 @ R3 @ R4 @ R5

    # 위치와 회전 행렬 반환
    return end_pos, R_end

# --- 역기구학 계산에 사용할 오차 함수 ---
# IK는 목표 위치와 목표 자세를 만족하는 관절각을 찾는 문제인데,
# 이때 오차함수는 현재 FK 결과와 목표 위치, 목표 자세 차이의 크기를 나타냅니다.

def fk_error(q_deg, target_pos, target_rot):
    # 입력 관절각 q_deg로 FK 계산
    pos, rot = fk_5dof_shoulder_custom(*q_deg)
    # 위치 오차는 두 위치 벡터 차이의 유클리드 거리 (크기)
    pos_err = np.linalg.norm(pos - target_pos)
    # 자세 오차는 두 회전 행렬 차이의 Frobenius norm (행렬 요소 차이 합)
    rot_err = np.linalg.norm(rot - target_rot)
    # 위치 오차와 자세 오차를 합산한 값 반환
    return pos_err + rot_err

# --- 수치 최적화를 이용한 IK 함수 ---
# scipy.optimize.minimize 함수를 사용해 오차 함수를 최소화하는
# 관절 각도를 찾아냅니다.

def ik_numeric(target_pos, target_rot, q_init=None):
    if q_init is None:
        # 초기값 없으면 모두 0도로 시작
        q_init = np.zeros(5)
    # BFGS 알고리즘 사용, 오차 함수 최소화 시도
    res = minimize(fk_error, q_init, args=(target_pos, target_rot), method='BFGS', options={'disp': False})
    # 최적화된 관절각 반환 (degree 단위)
    return res.x

# --- 메인 실행부 (테스트) ---
if __name__ == "__main__":
    # 임의 관절 각도 (degree)
    q_test = [45, -30, 20, 15, 10]
    # FK 실행 -> 위치와 회전 행렬 계산
    pos_fk, rot_fk = fk_5dof_shoulder_custom(*q_test)

    print("=== FK 결과 ===")
    print("위치 (m):", np.round(pos_fk, 5))
    print("회전 행렬:\n", np.round(rot_fk, 5))

    # IK 수치 최적화 실행 (초기값은 FK에서 쓴 각도)
    q_ik = ik_numeric(pos_fk, rot_fk, q_init=q_test)
    print("\n=== IK 결과 (수치 최적화) ===")
    print("계산된 관절 각도 (deg):", np.round(q_ik, 3))

    # IK->FK 재검산 (역기구학 결과를 다시 FK에 넣어서 위치 확인)
    pos_check, rot_check = fk_5dof_shoulder_custom(*q_ik)
    print("\n=== IK->FK 검산 ===")
    print("위치 (m):", np.round(pos_check, 5))
    print("위치 오차 (m):", np.round(pos_check - pos_fk, 7))

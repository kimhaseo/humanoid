import numpy as np

# ===== 회전 행렬 유틸리티 함수 정의 =====
def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def to_homogeneous(R, t):
    # 3x3 회전행렬 R과 3x1 위치벡터 t를 받아 4x4 동차변환행렬로 변환
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

# ===== 정방향 기구학 (Forward Kinematics) =====
def forward_kinematics(theta1, theta2, theta3, theta4, theta5, d1, d2, L1, L2, L3):
    L1_total = d2 + L1  # 어깨 yaw + 상완 길이 통합

    # 어깨 회전 3자유도
    T01 = to_homogeneous(rot_y(theta1), np.array([0, 0, d1]))              # θ1: Y축, 어깨 높이
    T12 = to_homogeneous(rot_x(theta2), np.array([0, 0, 0]))               # θ2: X축 회전만
    T23 = to_homogeneous(rot_z(theta3), np.array([0, 0, L1_total]))        # θ3: Z축 회전, 이동은 한꺼번에

    # 팔꿈치 + 손목
    T34 = to_homogeneous(rot_y(theta4), np.array([0, 0, L2]))              # θ4: Y축
    T45 = to_homogeneous(rot_y(theta5), np.array([0, 0, L3]))              # θ5: Y축

    # 누적변환
    T05 = T01 @ T12 @ T23 @ T34 @ T45
    return T05
# ===== 역기구학 (Inverse Kinematics) =====
def inverse_kinematics(T06, d1, d2, L1, L2, L3):
    # 목표 위치와 자세 행렬 분리
    R06 = T06[:3,:3]           # 최종 회전행렬
    P06 = T06[:3,3]            # 최종 위치벡터

    # Step 1: 손끝에서 L3만큼 뒤로 빼 손목 중심 위치 구함
    P_wc = P06 - R06[:,2] * L3  # 엔드이펙터 Z축 방향으로 L3만큼 빼기
    x_wc, y_wc, z_wc = P_wc

    x_wc = 100
    y_wc = 40
    z_wc = 100

    # Step 2: 어깨 Y축 회전 각도 θ1 (x-z 평면 기준)
    theta1 = np.arctan2(x_wc, z_wc)  # 주의: arctan2(x, z)

    # Step 3: 어깨 기준으로 수평거리 r, 수직거리 y 구함
    r = np.sqrt(x_wc**2 + z_wc**2)
    y = y_wc - d1   # d1 + d2는 어깨에서 팔꿈치까지 오프셋

    # Step 4: 팔꿈치 각도 θ4 계산 (코사인 법칙)
    L1_total = d2 + L1  # 전체 상완 거리 (d2 포함)
    D = (L1_total ** 2 + L2 ** 2 - (r ** 2 + y ** 2)) / (2 * L1_total * L2)
    D = np.clip(D, -1.0, 1.0)
    theta4 = np.arccos(D)

    # Step 5: 어깨 X축 회전 각도 θ2 계산
    phi = np.arctan2(y, r)  # 어깨와 손목 사이 각도
    psi = np.arctan2(L2*np.sin(theta4), L1 + L2*np.cos(theta4))  # 삼각형 내부각
    theta2 = phi - psi

    # Step 6: T04 계산 (어깨~팔꿈치 끝까지)
    L1_total = d2 + L1  # 상완 전체 거리

    T01 = to_homogeneous(rot_y(theta1), np.array([0, 0, d1]))
    T12 = to_homogeneous(rot_x(theta2), np.array([0, 0, 0]))  # X축 회전만
    T23 = to_homogeneous(rot_z(0), np.array([0, 0, L1_total]))  # Z축 회전, 이동 포함
    T34 = to_homogeneous(rot_y(theta4), np.array([0, 0, 0]))  # Y축 회전만

    T04 = T01 @ T12 @ T23 @ T34
    R04 = T04[:3, :3]

    # Step 7: 팔꿈치 이후의 회전행렬 R46 구함
    R46 = R04.T @ R06

    # Step 8: 어깨 Z축 회전 θ3 계산 (좌우로 비트는 회전)
    theta3 = np.arctan2(R46[1,0], R46[0,0])

    # Step 9: 손목 pitch 회전 θ5 계산 (Y축 회전)
    theta5 = np.arctan2(-R46[2,0], R46[2,2])

    return theta1, theta2, theta3, theta4, theta5

# ===== 테스트 예시 =====
px, py, pz = 100, 40, 70  # 목표 위치
R_desired = rot_z(np.radians(0)) @ rot_y(0) @ rot_x(0)  # 목표 자세 (회전 행렬)
T06 = to_homogeneous(R_desired, np.array([px, py, pz]))

# 로봇 링크 파라미터
d1 = 40   # 어깨에서 상체 시작 높이
d2 = 50   # 어깨 두께 또는 offset
L1 = 100  # 상완 길이
L2 = 100  # 하완 길이
L3 = 50   # 손끝 길이

# ===== 역기구학 수행 =====
angles = inverse_kinematics(T06, d1, d2, L1, L2, L3)
print("관절 각도 (도):", np.degrees(angles))

# ===== 정방향 검증 =====
T05 = forward_kinematics(*angles, d1, d2, L1, L2, L3)
print("계산된 엔드이펙터 위치:", T05[:3,3])
print("목표 엔드이펙터 위치:", T06[:3,3])

## 공부 중
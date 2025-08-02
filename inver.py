import numpy as np


# --- 유틸리티 함수 ---

def transformation_matrix(theta, d, a, alpha):
    """
    DH(Denavit-Hartenberg) 파라미터를 기반으로 단일 관절의 변환 행렬을 생성합니다.
    이 4x4 행렬은 한 좌표계에서 다음 좌표계로의 변환(회전 및 이동)을 나타냅니다.

    Args:
        theta (float): z축 기준 회전 각도 (degrees)
        d (float): z축 기준 이동 거리
        a (float): x축 기준 이동 거리 (링크 길이)
        alpha (float): x축 기준 회전 각도 (링크 비틀림)

    Returns:
        np.array: 4x4 변환 행렬
    """
    # 계산을 위해 각도를 라디안으로 변환합니다.
    theta_rad = np.radians(theta)
    alpha_rad = np.radians(alpha)

    ct = np.cos(theta_rad)
    st = np.sin(theta_rad)
    ca = np.cos(alpha_rad)
    sa = np.sin(alpha_rad)

    # 표준 DH 변환 행렬 공식
    T = np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    return T


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    사람에게 직관적인 오일러 각(Roll, Pitch, Yaw)을 3x3 회전 행렬로 변환합니다.
    회전 순서는 Z-Y-X 순서를 따릅니다. (Yaw -> Pitch -> Roll)

    Args:
        roll (float): x축 회전 (라디안)
        pitch (float): y축 회전 (라디안)
        yaw (float): z축 회전 (라디안)

    Returns:
        np.array: 3x3 회전 행렬
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Z-Y-X 순서로 행렬을 곱하여 최종 회전 행렬을 만듭니다.
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# --- 순기구학 (Forward Kinematics) ---

def forward_kinematics(dh_params, angles_deg):
    """
    순기구학: 주어진 모든 관절의 각도를 바탕으로 로봇팔 끝의 최종 위치와 자세를 계산합니다.
    역기구학으로 구한 해가 맞는지 검증하는 데 사용됩니다.

    Args:
        dh_params (list): 로봇의 전체 DH 파라미터 리스트
        angles_deg (list): 6개 관절의 각도 (degrees)

    Returns:
        np.array: 로봇팔 끝의 위치와 자세를 나타내는 4x4 변환 행렬
    """
    # 베이스 좌표계를 기준으로 시작 (단위 행렬)
    T_total = np.identity(4)
    # 각 관절의 변환 행렬을 순서대로 계속 곱해나갑니다.
    # T_0_6 = T_0_1 * T_1_2 * ... * T_5_6
    for i in range(len(dh_params)):
        theta, d, a, alpha = dh_params[i]
        # DH 테이블의 theta는 고정된 오프셋 값이고, angles_deg[i]가 실제 움직이는 변수입니다.
        T_i = transformation_matrix(angles_deg[i] + theta, d, a, alpha)
        T_total = np.dot(T_total, T_i)
    return T_total


# --- 역기구학 (Inverse Kinematics) ---

def inverse_kinematics(dh_params, target_pose, elbow_config='down'):
    """
    역기구학: 목표 위치와 자세가 주어졌을 때, 6개 관절의 각도를 계산합니다.
    '기구학적 분리(Kinematic Decoupling)'라는 표준 해법을 사용합니다.
    이 방법은 복잡한 6자유도 문제를 두 개의 간단한 3자유도 문제로 나누어 풉니다.
    - 문제 1: 손목 중심의 '위치' 맞추기 (관절 1, 2, 3)
    - 문제 2: 손목의 '자세' 맞추기 (관절 4, 5, 6)
    """
    # === DH 파라미터에서 필요한 링크 길이들을 미리 추출합니다. ===
    d1 = dh_params[0][1]  # 베이스에서 관절 2까지의 수직 거리
    a2 = dh_params[1][2]  # 관절 2에서 관절 3까지의 링크(위팔) 길이
    d4 = dh_params[3][1]  # 관절 3에서 관절 5까지의 링크(아래팔) 길이
    d6 = dh_params[5][1]  # 손목 중심에서 로봇팔 끝(End-effector)까지의 거리

    # === 1. 손목 중심(Wrist Center, P_wc) 위치 계산 ===
    # P_wc는 관절 4, 5, 6이 만나는 한 점입니다. 이 점의 위치를 먼저 찾습니다.
    R_target = target_pose[:3, :3]  # 목표 자세 (3x3 회전 행렬)
    P_target = target_pose[:3, 3]  # 목표 위치 (3x1 벡터)

    # P_wc 계산 공식: P_wc = P_target - d6 * z_axis_of_hand
    # 의미: "목표 지점(P_target)에서 손의 Z축 방향(R_target[:, 2])으로
    #        손 길이(d6)만큼 거슬러 올라가면, 그곳이 바로 손목 중심(P_wc)이다."
    P_wc = P_target - d6 * R_target[:, 2]
    xc, yc, zc = P_wc

    # === 2. 처음 3개 관절 (q1, q2, q3) 계산 - 위치 문제 ===
    # 이제 P_wc의 좌표 (xc, yc, zc)를 이용해 q1, q2, q3를 구합니다.

    # q1 계산 (베이스 회전 각도)
    # 로봇을 위에서 내려다봤을 때, P_wc를 향하기 위한 각도입니다.
    q1 = np.arctan2(yc, xc)

    # --- q2, q3 계산 (2D 평면 문제로 변환) ---
    # q1이 정해지면, 문제는 q1 각도의 2D 평면에서 팔을 뻗는 문제가 됩니다.
    # r: 베이스 중심에서 P_wc까지의 수평 거리
    # S: 베이스 위(관절 1 높이)에서 P_wc까지의 수직 높이
    r = np.sqrt(xc ** 2 + yc ** 2)
    S = zc - d1

    # L_sq: 관절 2에서 손목 중심(P_wc)까지의 직선 거리의 '제곱'
    L_sq = r ** 2 + S ** 2

    # q3 계산 (코사인 제2법칙 사용)
    # 관절 2, 관절 3(팔꿈치), P_wc가 이루는 삼각형을 생각합니다.
    # 삼각형의 세 변의 길이는 a2, d4, L(sqrt(L_sq)) 입니다.
    # 코사인 법칙: L^2 = a2^2 + d4^2 - 2*a2*d4*cos(pi - q3_internal)
    # cos(pi - x) = -cos(x) 이므로, L^2 = a2^2 + d4^2 + 2*a2*d4*cos(q3_internal)
    # 이 식을 변형하면 아래와 같이 cos(q3)를 구할 수 있습니다.
    # (여기서 q3는 DH 정의에 따른 각도이며, 삼각형 내각과 부호가 다를 수 있습니다)
    cos_q3 = (L_sq - a2 ** 2 - d4 ** 2) / (2 * a2 * d4)

    # cos_q3의 절대값이 1보다 크면 물리적으로 도달 불가능한 지점입니다.
    if abs(cos_q3) > 1:
        print("오류: 목표 지점에 도달할 수 없습니다. (너무 멀거나 가까움)")
        return None

    # 팔꿈치 방향(up/down)에 따라 q3의 해가 두 개 나옵니다.
    if elbow_config == 'up':
        sin_q3 = -np.sqrt(1 - cos_q3 ** 2)  # Elbow up
    else:  # 'down'
        sin_q3 = np.sqrt(1 - cos_q3 ** 2)  # Elbow down
    q3 = np.arctan2(sin_q3, cos_q3)

    # q2 계산 (alpha-beta 접근법)
    # q2는 두 개의 각도(alpha, beta)의 합 또는 차로 구합니다.
    # alpha: 관절 2와 P_wc를 잇는 선이 수평선과 이루는 각도
    alpha = np.arctan2(S, r)
    # beta: 관절 2-팔꿈치-P_wc 삼각형에서, 관절 2 쪽의 내각
    # 이 역시 코사인 법칙으로 구합니다.
    cos_beta = (L_sq + a2 ** 2 - d4 ** 2) / (2 * a2 * np.sqrt(L_sq))
    if abs(cos_beta) > 1: cos_beta = np.sign(cos_beta)  # 부동소수점 오차 방지
    sin_beta = np.sqrt(1 - cos_beta ** 2)
    beta = np.arctan2(sin_beta, cos_beta)

    # 팔꿈치 방향에 따라 alpha와 beta를 더하거나 뺍니다.
    if elbow_config == 'up':
        q2 = alpha + beta
    else:  # 'down'
        q2 = alpha - beta

    # === 3. 마지막 3개 관절 (q4, q5, q6) 계산 - 자세 문제 ===
    # 이제 손목의 위치를 맞췄으니, 손목을 회전시켜 목표 자세를 맞춥니다.
    # 핵심 아이디어: R_0_6 = R_0_3 * R_3_6
    # R_0_6는 목표 자세(R_target)이고, R_0_3는 q1,q2,q3로 계산 가능합니다.
    # 따라서 R_3_6 = (R_0_3)^-1 * R_0_6 를 계산하여 q4,q5,q6를 구합니다.

    # 먼저 R_0_3를 계산합니다.
    T0_1 = transformation_matrix(np.degrees(q1) + dh_params[0][0], dh_params[0][1], dh_params[0][2], dh_params[0][3])
    T1_2 = transformation_matrix(np.degrees(q2) + dh_params[1][0], dh_params[1][1], dh_params[1][2], dh_params[1][3])
    T2_3 = transformation_matrix(np.degrees(q3) + dh_params[2][0], dh_params[2][1], dh_params[2][2], dh_params[2][3])
    T0_3 = T0_1 @ T1_2 @ T2_3
    R0_3 = T0_3[:3, :3]

    # R3_6 계산. 회전 행렬의 역행렬은 전치 행렬과 같습니다. (R^-1 = R^T)
    R3_6 = R0_3.T @ R_target

    # R3_6 행렬의 원소들로부터 q4, q5, q6를 추출합니다.
    # 특이점(Singularity, Gimbal Lock) 처리: q5가 ±90도일 때
    if np.isclose(R3_6[2, 2], -1):  # sin(q5) = -1, q5 = 90도
        q5 = np.pi / 2
        # q4와 q6의 차이만 의미 있으므로, q6=0으로 두고 q4를 계산
        q4 = np.arctan2(R3_6[1, 0], R3_6[0, 0])
        q6 = 0
    elif np.isclose(R3_6[2, 2], 1):  # sin(q5) = 1, q5 = -90도
        q5 = -np.pi / 2
        q4 = np.arctan2(-R3_6[1, 0], -R3_6[0, 0])
        q6 = 0
    else:  # 일반적인 경우
        q4 = np.arctan2(R3_6[1, 2], R3_6[0, 2])
        q5 = np.arctan2(np.sqrt(R3_6[0, 2] ** 2 + R3_6[1, 2] ** 2), R3_6[2, 2])
        q6 = np.arctan2(R3_6[2, 1], -R3_6[2, 0])

    # 최종 계산된 각도들을 라디안에서 degree로 변환하여 반환
    angles_rad = np.array([q1, q2, q3, q4, q5, q6])
    return np.degrees(angles_rad)


# --- 메인 실행 블록 ---
if __name__ == '__main__':
    # === 로봇팔의 물리적 구조를 DH 파라미터로 정의합니다. ===
    # [theta, d, a, alpha] (단위: m, degrees)
    # 이 값들은 실제 제작할 로봇의 설계에 따라 정확하게 수정해야 합니다.
    DH_PARAMS = [
        [0, 0.3, 0.0, 90],  # Link 1 -> 2
        [0, 0.0, 0.25, 0],  # Link 2 -> 3
        [90, 0.0, 0.0, 90],  # Link 3 -> 4
        [0, 0.2, 0.0, -90],  # Link 4 -> 5
        [0, 0.0, 0.0, 90],  # Link 5 -> 6
        [0, 0.1, 0.0, 0]  # Link 6 -> End-effector
    ]

    # === 목표 위치 및 자세를 설정합니다. ===
    target_position = [0.2, 0.15, 0.3]
    target_orientation_rad = [np.pi / 2, 0, np.pi / 4]  # Roll, Pitch, Yaw (라디안)

    # 목표 변환 행렬(4x4)을 생성합니다.
    R_target = euler_to_rotation_matrix(*target_orientation_rad)
    P_target = np.array(target_position)
    TARGET_POSE = np.identity(4)
    TARGET_POSE[:3, :3] = R_target
    TARGET_POSE[:3, 3] = P_target

    print("--- 목표 포즈 (위치+자세) ---")
    print(np.round(TARGET_POSE, 3))
    print("\n")

    # === 역기구학을 계산합니다. ===
    print("--- 역기구학 계산 결과 ('Down' Elbow) ---")
    ik_angles = inverse_kinematics(DH_PARAMS, TARGET_POSE, elbow_config='down')

    # === 결과를 검증합니다. ===
    if ik_angles is not None:
        print(f"계산된 관절 각도 (q1~q6):")
        print(f"{np.round(ik_angles, 2)} (degrees)\n")

        # 계산된 각도를 순기구학에 넣어 원래 목표와 비교합니다.
        print("--- 순기구학 검증 ---")
        fk_pose = forward_kinematics(DH_PARAMS, ik_angles)
        print("계산된 각도로부터 얻은 최종 포즈:")
        print(np.round(fk_pose, 3))

        # 목표와 결과 사이의 오차를 계산합니다. 이 값이 0에 가까울수록 정확합니다.
        pos_error = np.linalg.norm(TARGET_POSE[:3, 3] - fk_pose[:3, 3])
        ori_error = np.linalg.norm(TARGET_POSE[:3, :3] - fk_pose[:3, :3])
        print(f"\n목표 위치와의 오차: {pos_error:.6f} m")
        print(f"목표 자세와의 오차: {ori_error:.6f}")
import numpy as np
import warnings

# 경고 메시지를 무시하여 atan2 결과가 복소수 경고를 피하도록 설정
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --- 순기구학 (FK) 기본 함수 ---

def dh_matrix(a, alpha, d, theta):
    """Denavit-Hartenberg 변환 행렬 생성 (회전 관절 기준)."""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # D-H 행렬
    T = np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0, sa, ca, d],
        [0, 0, 0, 1]
    ])
    return T


# --- 6-DOF 역기구학 함수 (옵셋 고려) ---

def inverse_kinematics_6dof_offset(T_target, dh_params):
    """
    6-DOF 로봇팔의 역기구학을 계산합니다. (d2 옵셋 고려)
    T_target: 목표 말단 (End-Effector) 변환 행렬 (4x4)
    dh_params: [a_i, alpha_i, d_i] 튜플 리스트 (Link 1부터 6까지)

    반환값: 6개의 관절 각도 (라디안) [theta1, ..., theta6] (여러 해 중 하나)
    """

    # D-H 매개변수 추출
    L = dh_params

    # 로봇 길이 상수
    A2 = L[1][0]  # Link 2의 길이 a2
    A3 = L[2][0]  # Link 3의 길이 a3
    D1 = L[0][2]  # Link 1의 옵셋 d1
    D2 = L[1][2]  # Link 2의 옵셋 d2 (어깨 옵셋)
    D6 = L[5][2]  # Link 6의 옵셋 d6 (손목 중심점 계산에 사용)

    # 목표 위치/자세 추출
    P_t = T_target[:3, 3]  # 목표 위치 [px, py, pz]
    R_t = T_target[:3, :3]  # 목표 자세 행렬

    # --- A. 암 관절 (theta1, theta2, theta3) 계산 ---

    # 1. 손목 중심점 (Wrist Center, P_w) 계산
    # P_w = P_t - D6 * Z_6_hat
    Z_6_hat = R_t[:, 2]
    P_w = P_t - D6 * Z_6_hat
    Pxw, Pyw, Pzw = P_w[0], P_w[1], P_w[2]

    # 2. Theta 1 계산 (어깨 옵셋 D2 고려)
    # Pxw, Pyw를 이용해 r을 구하고, r, D2, L_w를 이용해 theta1을 계산

    r_sq = Pxw ** 2 + Pyw ** 2
    r = np.sqrt(r_sq)

    if r < np.abs(D2):
        print("Error: Target too close, cannot satisfy d2 offset.")
        return None

    # 보조 각도 phi (P_w의 x-y 평면 투영 각도)
    phi = np.arctan2(Pyw, Pxw)

    # 보조 각도 alpha (r, D2를 이용한 삼각형의 각도)
    try:
        cos_alpha = D2 / r
        alpha = np.arccos(np.clip(cos_alpha, -1, 1))
    except ValueError:
        print("Error: D2 is too large for the current target distance.")
        return None

    # Theta 1 해 (두 가지 해 중 하나, 예: 팔꿈치 앞으로/뒤로에 따라)
    # theta1 = phi + alpha (Solution 1)
    theta1 = phi - alpha  # Solution 2 (여기서는 이 해를 사용)

    # 3. P_w를 1번 관절 좌표계(Frame 1)로 변환
    # T0_1 역행렬을 이용해야 하지만, 단순화를 위해 회전만 고려

    # C1 = cos(theta1), S1 = sin(theta1)
    C1 = np.cos(theta1)
    S1 = np.sin(theta1)

    # P_w_1 = R0_1.T * P_w - R0_1.T * P0_1
    # P_w_1: Frame 1에서의 손목 중심 위치

    # x1w = C1 * Pxw + S1 * Pyw
    # y1w = -S1 * Pxw + C1 * Pyw
    # z1w = Pzw - D1 (z축 옵셋 처리)

    x1w = C1 * Pxw + S1 * Pyw
    z1w = Pzw - D1  # Pzw는 베이스 기준, D1은 1번 축 옵셋

    # 4. Theta 3 계산 (코사인 법칙)
    # 이제 x1w와 z1w를 이용하여 2차원 평면 IK처럼 풀 수 있습니다.

    r_xz_sq = x1w ** 2 + z1w ** 2
    r_xz = np.sqrt(r_xz_sq)

    # L_sq는 A2와 A3가 이루는 삼각형의 밑변 길이의 제곱
    L_sq = r_xz_sq - D2 ** 2  # D2 옵셋을 이미 처리했으므로 r_xz_sq는 삼각형의 빗변이 됨
    L = np.sqrt(L_sq)

    # 코사인 법칙: cos(phi3) = (L^2 - A2^2 - A3^2) / (2 * A2 * A3)
    try:
        cos_phi3 = (L_sq - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)

        phi3 = np.arccos(np.clip(cos_phi3, -1, 1))
        theta3 = phi3 - np.pi  # Elbow Up Solution 가정
        # theta3 = np.pi - phi3 # Elbow Down Solution

    except ValueError:
        print("Error: Target out of reach (Non-real solution for theta3)")
        return None

    # 5. Theta 2 계산

    # 보조 각도 beta (r_xz와 z1w를 이용한 각도)
    beta = np.arctan2(z1w, x1w)

    # 보조 각도 gamma (A2, A3, L을 이용한 삼각형 내부 각도)
    # 코사인 법칙: cos(gamma) = (A2^2 + L^2 - A3^2) / (2 * A2 * L)
    cos_gamma = (A2 ** 2 + L_sq - A3 ** 2) / (2 * A2 * L)
    gamma = np.arccos(np.clip(cos_gamma, -1, 1))

    # theta2 (Elbow Up)
    theta2 = beta + gamma
    # theta2 = beta - gamma # Elbow Down 해

    # --- B. 손목 관절 (theta4, theta5, theta6) 계산 ---

    # 6. R3_6 (R_target) 계산

    # T0_3 (순기구학) 계산
    T0_1 = dh_matrix(L[0][0], L[0][1], L[0][2], theta1)
    T1_2 = dh_matrix(L[1][0], L[1][1], L[1][2], theta2)
    T2_3 = dh_matrix(L[2][0], L[2][1], L[2][2], theta3)
    T0_3 = T0_1 @ T1_2 @ T2_3

    R0_3 = T0_3[:3, :3]
    R3_6 = R0_3.T @ R_t  # R0_3.T는 R0_3의 역행렬

    # R3_6 행렬 성분 추출 (자세 계산)
    r43, r53, r63 = R3_6[:, 2]  # 3열
    r42, r52, r62 = R3_6[:, 1]  # 2열
    r41, r51, r61 = R3_6[:, 0]  # 1열

    # 7. Theta 5 계산
    # theta5 = atan2( +-sqrt(r43^2 + r53^2), r63 )
    sin_theta5 = np.sqrt(r43 ** 2 + r53 ** 2)
    theta5 = np.arctan2(sin_theta5, r63)  # (+ sin_theta5 해)

    # 8. Theta 4, Theta 6 계산 (r63 != 1 일 때)
    if np.isclose(sin_theta5, 0):
        # 특이점 (Wrist Singularity): theta4와 theta6 합이 결정됨
        theta4 = 0.0  # 임의 설정
        theta6 = np.arctan2(-r42, r41)  # R_z(theta6) * R_z(theta4) = R_z(theta4+theta6)
    else:
        # 분모 sin(theta5)로 나누어 계산
        theta4 = np.arctan2(r53 / sin_theta5, -r43 / sin_theta5)
        theta6 = np.arctan2(-r62 / sin_theta5, r61 / sin_theta5)

    return np.array([theta1, theta2, theta3, theta4, theta5, theta6])
import numpy as np
from scipy.optimize import least_squares

# ---------------------------
# 귀하의 원래 회전행렬 (정의 그대로 사용)
# ---------------------------
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

# ---------------------------
# 링크 길이 (m) — 귀하 정의와 동일하게 유지
# ---------------------------
d1 = 0.05
L2 = 0.06
L3 = 0.06
L4 = 0.07
L5 = 0.05

# ---------------------------
# 귀하의 원래 FK 함수(degree 입력) — 그대로 사용
# 반환: (pos (3,), R_end (3x3))
# ---------------------------
def fk_5dof_shoulder_custom(q1_deg, q2_deg, q3_deg, q4_deg, q5_deg):
    q1 = np.radians(q1_deg)
    q2 = np.radians(q2_deg)
    q3 = np.radians(q3_deg)
    q4 = np.radians(q4_deg)
    q5 = np.radians(q5_deg)

    R1 = rotation_y(q1)
    R2 = rotation_x(q2)
    R3 = rotation_z(q3)
    R4 = rotation_y(q4)
    R5 = rotation_y(q5)

    offset_joint2 = np.array([0, d1, 0])
    pos_joint2 = R1 @ offset_joint2
    link2_vector = np.array([0, 0, -L2])
    pos_joint3 = pos_joint2 + R1 @ R2 @ link2_vector
    link3_vector = np.array([0, 0, -L3])
    pos_joint4 = pos_joint3 + R1 @ R2 @ R3 @ link3_vector
    link4_vector = np.array([L4, 0, 0])
    pos_joint5 = pos_joint4 + R1 @ R2 @ R3 @ R4 @ link4_vector
    link5_vector = np.array([L5, 0, 0])
    end_pos = pos_joint5 + R1 @ R2 @ R3 @ R4 @ R5 @ link5_vector

    R_end = R1 @ R2 @ R3 @ R4 @ R5
    return end_pos, R_end

# ---------------------------
# 회전행렬 -> so(3) 벡터 (axis * angle) 변환 (작은 각 근처 안전 처리 포함)
# 입력: R (3x3), 반환: 3-vector (rad)
# 이 함수는 R이 identity일 때 [0,0,0] 리턴
# ---------------------------
def rotation_matrix_to_axis_angle(R):
    # trace
    tr = np.trace(R)
    # numerical safety
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if np.isclose(theta, 0.0):
        return np.zeros(3)
    # axis (from skew-symmetric part)
    rx = (R[2,1] - R[1,2]) / (2*np.sin(theta))
    ry = (R[0,2] - R[2,0]) / (2*np.sin(theta))
    rz = (R[1,0] - R[0,1]) / (2*np.sin(theta))
    axis = np.array([rx, ry, rz])
    return axis * theta

# ---------------------------
# 잔차 함수: 입력은 angle vector (degrees)
# 반환: 6-vector (pos_res(3), rot_res(3))
# target_rot, target_pos는 world frame
# ---------------------------
def residuals_deg(q_deg, target_pos, target_rot):
    # q_deg: length-5 array (degrees)
    pos_fk, R_fk = fk_5dof_shoulder_custom(*q_deg)
    # position residuals
    pos_res = pos_fk - target_pos
    # rotation residuals: R_err = R_fk^T * R_target  or R_target^T * R_fk?
    # we want R_fk -> R_target, so R_err = R_target @ R_fk^T
    R_err = target_rot @ R_fk.T
    rot_res = rotation_matrix_to_axis_angle(R_err)  # 3-vector (rad)
    # scale rotation residual to same units as position (optional tuning)
    # here we return raw (pos in meters, rot in radians)
    return np.hstack((pos_res, rot_res))

# ---------------------------
# numeric IK using least_squares
# - q_init: degrees initial guess (len 5) optional
# - returns: optimized q (degrees), success flag, result object
# ---------------------------
def ik_numeric_least_squares(target_pos, target_rot, q_init_deg=None, verbose=False):
    if q_init_deg is None:
        q_init_deg = np.array([0., 0., 0., 0., 0.])
    else:
        q_init_deg = np.array(q_init_deg, dtype=float)

    # least_squares minimize residuals
    res = least_squares(
        fun=lambda q: residuals_deg(q, target_pos, target_rot),
        x0=q_init_deg,
        method='lm',   # Levenberg-Marquardt (no bounds). If you want bounds, use 'trf' or 'dogbox'.
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-8,
        max_nfev=2000,
        verbose=2 if verbose else 0
    )
    return res.x, res.success, res

# ---------------------------
# 메인: 테스트 및 예시 (귀하가 올렸던 목표값으로 실행 예시)
# ---------------------------
if __name__ == "__main__":
    # 귀하가 올려주신 FK 결과(예시) — 이 값으로 IK를 풀어봄
    target_pos = np.array([-0.05,  0.03, -0.2])   # m
    target_rot = np.array([[ 0.246428, -0.469105,  0.848065],
                           [ 0.146738,  0.883022,  0.445802],
                           [-0.957988,  0.014585,  0.286437]])

    # 초기값: 원래 코드에서 썼던 각도(있다면) 또는 0으로 시작
    q_init = [45, -20, 20, 15, 10]  # degree (try a reasonable guess)
    q_sol_deg, ok, res_obj = ik_numeric_least_squares(target_pos, target_rot, q_init_deg=q_init, verbose=True)

    print("\n=== 수치 IK 결과 ===")
    print("성공:", ok)
    print("계산된 관절 각도 (deg):", np.round(q_sol_deg, 6))

    # 검산
    pos_check, rot_check = fk_5dof_shoulder_custom(*q_sol_deg)
    print("\n=== IK->FK 검산 ===")
    print("재계산 위치 (m):", np.round(pos_check, 3))
    print("목표 위치 (m):", np.round(target_pos, 3))
    print("위치 오차 (m):", np.round(pos_check - target_pos, 3))
    print("자세 오차 (axis-angle norm, rad):", np.round(np.linalg.norm(rotation_matrix_to_axis_angle(target_rot @ rot_check.T)),3))


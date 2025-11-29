import numpy as np
import math


# --- 1. 회전 행렬 함수 ---
def R_x(theta):
    """X축 기준 회전 행렬"""
    c = math.cos(theta);
    s = math.sin(theta)
    return np.array([
        [1, 0, 0], [0, c, -s], [0, s, c]
    ])


def R_y(theta):
    """Y축 기준 회전 행렬"""
    c = math.cos(theta);
    s = math.sin(theta)
    return np.array([
        [c, 0, s], [0, 1, 0], [-s, 0, c]
    ])


def R_z(theta):
    """Z축 기준 회전 행렬"""
    c = math.cos(theta);
    s = math.sin(theta)
    return np.array([
        [c, -s, 0], [s, c, 0], [0, 0, 1]
    ])


# --- 2. 정기구학(FK) 함수 (라디안 입력) ---
def shoulder_5dof_fk_rad(q_rad,
                         shoulder=(0.0, 0.0, 0.0),
                         d_y=0.05, d_z=-0.05, d4_z=0.05, d5_z=0.05, ee_z=0.03):
    """
    수정된 5DOF 정기구학 (FK) - 라디안 입력
    """
    O = np.array(shoulder, dtype=float)
    O2 = O + np.array([0, d_y, 0])
    R1 = R_y(q_rad[0])
    O3_offset = np.array([0, 0, d_z])
    O3 = O2 + O3_offset
    R2 = R1 @ R_x(q_rad[1])
    O4_local_offset = np.array([0, 0, -d4_z])
    O4 = O3 + R2 @ O4_local_offset
    R3 = R2 @ R_z(q_rad[2])
    O5_local_offset = np.array([0, 0, -d5_z])
    O5 = O4 + R3 @ O5_local_offset
    R4 = R3 @ R_y(q_rad[3])
    EE_local_offset = np.array([0, 0, -ee_z])
    EE = O5 + R4 @ EE_local_offset
    return EE


# ----------------------------------------------------
# --- 3. 자코비안 및 안정화된 IK 솔버 함수 ---
# ----------------------------------------------------

def calculate_jacobian(q_rad, fk_func, epsilon=1e-6):
    """
    수치적 자코비안 행렬 (J) 계산
    """
    n_joints = len(q_rad)
    J = np.zeros((3, n_joints))
    P_current = fk_func(q_rad)

    for i in range(n_joints):
        q_perturbed = np.copy(q_rad)
        q_perturbed[i] += epsilon
        P_perturbed = fk_func(q_perturbed)
        J[:, i] = (P_perturbed - P_current) / epsilon

    return J


def shoulder_5dof_ik_solver_stable(Px, Py, Pz, q_start_deg,
                                   max_iterations=10000, tolerance=1e-5, learning_rate=0.03):
    """
    자코비안 기반의 안정화된 역기구학 솔버 (Damping 적용)
    """

    P_target = np.array([Px, Py, Pz])
    q_rad = np.radians(np.array(q_start_deg, dtype=float))

    print(f"IK 계산 시작. 목표: ({Px}, {Py}, {Pz}) (수정된 매개변수 적용)")

    for i in range(max_iterations):
        P_current = shoulder_5dof_fk_rad(q_rad)
        error = P_target - P_current

        # 종료 조건 확인
        error_norm = np.linalg.norm(error)
        if error_norm < tolerance:
            print(f"IK 성공! 반복 횟수: {i}회, 최종 오차: {error_norm:.7f}m")
            return np.degrees(q_rad)

        # 자코비안 (J) 및 의사 역행렬 (J_pinv) 계산
        J = calculate_jacobian(q_rad, shoulder_5dof_fk_rad)
        J_pinv = np.linalg.pinv(J)

        # 5. 관절 각도 업데이트
        delta_q = J_pinv @ error * learning_rate

        # --- 안전 장치: 최대 각도 변화량 제한 (Damping) ---
        max_delta_q = np.radians(5.0)  # 최대 5도로 제한
        delta_q_norm = np.linalg.norm(delta_q)

        if delta_q_norm > max_delta_q:
            delta_q = delta_q * (max_delta_q / delta_q_norm)
        # --------------------------------------------------------

        q_rad += delta_q

    # 실패 시
    error_norm = np.linalg.norm(error)
    print(f"IK 실패! 최대 반복 횟수 도달. 최종 오차: {error_norm:.7f}m")
    return np.degrees(q_rad)


# ----------------------------------------------------
# --- 4. 최종 실행 및 검증 (수정된 매개변수) ---
# ----------------------------------------------------

# 목표 위치: (0.00, 0.05, 0.18)
target_Px, target_Py, target_Pz = 0.00, 0.05, -0.18

# **수정된 초기 각도:** 특이점 탈출을 위해 작은 오프셋 적용
q_start = [5.0, 5.0, 5.0, 5.0, 5.0]

# IK 실행 (안정화 솔버 호출, learning_rate=0.03 적용)
q_solution_deg = shoulder_5dof_ik_solver_stable(
    target_Px, target_Py, target_Pz, q_start,
    learning_rate=0.03,
    max_iterations=10000
)

# --- 결과 출력 ---
if q_solution_deg is not None:
    print("\n" + "=" * 50)
    print(f"## 🏆 {target_Px, target_Py, target_Pz} 에 대한 IK 최종 해")
    print(f"q1~q5 (deg): {q_solution_deg}")

    # FK로 결과 검증
    P_target = np.array([target_Px, target_Py, target_Pz])
    EE_check = shoulder_5dof_fk_rad(np.radians(q_solution_deg))

    print("-" * 50)
    print(f"FK 검증 위치 (m): {EE_check}")
    print(f"목표 위치 (m): {P_target}")
    print(f"최종 위치 오차: {np.linalg.norm(EE_check - P_target):.7f}m")
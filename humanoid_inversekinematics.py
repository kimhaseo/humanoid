import numpy as np
from scipy.spatial.transform import Rotation


# ==============================================================================
# 1. 기본 함수 및 순기구학 (변경 없음)
# ==============================================================================
def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk_5dof_shoulder_custom(q1_deg, q2_deg, q3_deg, q4_deg, q5_deg):
    q = np.radians([q1_deg, q2_deg, q3_deg, q4_deg, q5_deg])
    q1, q2, q3, q4, q5 = q
    R1, R2, R3, R4, R5 = rotation_y(q1), rotation_x(q2), rotation_z(q3), rotation_y(q4), rotation_y(q5)
    offset_joint2 = np.array([0, 0.05, 0])
    link2_vector, link3_vector = np.array([0, 0, -0.06]), np.array([0, 0, -0.06])
    link4_vector, link5_vector = np.array([0.07, 0, 0]), np.array([0.05, 0, 0])
    T1, T1_2, T1_3, T1_4, T1_5 = R1, R1 @ R2, R1 @ R2 @ R3, R1 @ R2 @ R3 @ R4, R1 @ R2 @ R3 @ R4 @ R5
    pos_joint2 = T1 @ offset_joint2
    pos_joint3 = pos_joint2 + T1_2 @ link2_vector
    pos_joint4 = pos_joint3 + T1_3 @ link3_vector
    pos_joint5 = pos_joint4 + T1_4 @ link4_vector
    end_pos = pos_joint5 + T1_5 @ link5_vector
    R_end = T1_5
    return end_pos, R_end


# ==============================================================================
# 2. ✨ '피치'만 맞추는 새로운 역기구학 함수 ✨
# ==============================================================================
def get_joint_angles_for_pos_and_pitch(target_position, target_pitch_deg, initial_guess_deg=None):
    """
    목표 위치와 '피치' 자세만 맞추는 역기구학을 계산합니다.
    롤과 요는 해에 따라 자유롭게 결정됩니다.

    Args:
        target_position (list or np.array): 목표 위치 [x, y, z].
        target_pitch_deg (float): 목표 피치 각도 (도 단위).
        initial_guess_deg (list or np.array, optional): 초기 추정 관절 각도.

    Returns:
        np.array: 계산된 관절 각도 (도). 해를 찾지 못하면 None을 반환합니다.
    """
    print(f"목표 자세(Pitch={target_pitch_deg}°, Roll/Yaw=무관)에 대한 관절 각도를 계산합니다...")

    # --- IK 솔버 설정 ---
    if initial_guess_deg is None:
        q = np.zeros(5)
    else:
        q = np.radians(initial_guess_deg)

    max_iter, pos_tol, ori_tol, alpha, damping = 1000, 1e-5, 1e-4, 0.2, 0.02

    # --- IK 반복 계산 루프 ---
    for i in range(max_iter):
        # 1. 현재 관절 각도로 현재 위치/자세 계산 (FK)
        q_deg = np.degrees(q)
        current_pos, current_R = fk_5dof_shoulder_custom(*q_deg)

        # 2. 위치 오차 계산
        pos_error = np.array(target_position) - current_pos

        # --- ✨ 여기가 핵심 로직입니다 ✨ ---
        # 3. '동적 목표 자세' 생성
        # 현재 로봇의 자세를 RPY 각도로 변환 (순서: 요, 피치, 롤)
        current_rpy_deg = Rotation.from_matrix(current_R).as_euler('zyx', degrees=True)

        # 현재의 '요'와 '롤' 값은 그대로 사용하고, '피치'만 목표값으로 설정
        dynamic_target_rpy_deg = [current_rpy_deg[0], target_pitch_deg, current_rpy_deg[2]]

        # 이 동적 목표 자세를 다시 회전 행렬로 변환
        dynamic_target_R = Rotation.from_euler('zyx', dynamic_target_rpy_deg, degrees=True).as_matrix()
        # ------------------------------------

        # 4. 동적 목표를 기준으로 회전 오차 계산
        error_R = dynamic_target_R @ current_R.T
        ori_error = Rotation.from_matrix(error_R).as_rotvec()

        # 5. 최종 오차 벡터 결합
        error = np.concatenate([pos_error, ori_error])

        # 6. 수렴 조건 확인
        if np.linalg.norm(pos_error) < pos_tol and abs(ori_error[1]) < ori_tol:  # 피치 오차만 고려
            print(f"✅ 성공: {i + 1}번 반복 후 수렴했습니다.")
            return np.round(np.degrees(q), 4)

        # 7. 자코비안 계산 및 관절 각도 업데이트 (이전과 동일)
        R1, R2, R3, R4 = rotation_y(q[0]), rotation_x(q[1]), rotation_z(q[2]), rotation_y(q[3])
        T1, T1_2, T1_3, T1_4 = R1, R1 @ R2, R1 @ R2 @ R3, R1 @ R2 @ R3 @ R4
        p1, p2 = np.zeros(3), T1 @ np.array([0, 0.05, 0])
        p3 = p2 + T1_2 @ np.array([0, 0, -0.06])
        p4 = p3 + T1_3 @ np.array([0, 0, -0.06])
        p5 = p4 + T1_4 @ np.array([0.07, 0, 0])
        z1, z2, z3, z4, z5 = np.array([0, 1, 0]), T1 @ np.array([1, 0, 0]), T1_2 @ np.array([0, 0, 1]), T1_3 @ np.array(
            [0, 1, 0]), T1_4 @ np.array([0, 1, 0])
        J = np.array([np.concatenate([np.cross(z, current_pos - p), z]) for z, p in
                      [(z1, p1), (z2, p2), (z3, p3), (z4, p4), (z5, p5)]]).T
        delta_q = np.linalg.pinv(J, rcond=damping) @ error
        q += alpha * delta_q

    print(f"❌ 역기구학 해를 찾지 못했습니다. (최종 오차: {np.linalg.norm(error):.4f})")
    return None


# ==============================================================================
# 3. 메인 테스트 코드
# ==============================================================================
if __name__ == "__main__":

    # 테스트할 목표 위치와 '피치' 자세 설정
    pos_xyz = [0.12, 0.04, -0.08]  # x, y, z 위치 (m)
    pitch_deg = -0.0  # 목표 피치 자세 (도)

    print("-" * 50)
    print(f"목표 위치: {pos_xyz}")
    print(f"목표 자세 (피치): {pitch_deg}° (롤/요는 자유)")
    print("-" * 50)

    # 새로 만든 함수를 호출하여 관절 각도 계산
    ik_angles = get_joint_angles_for_pos_and_pitch(pos_xyz, pitch_deg)

    if ik_angles is not None:
        print("\n[결과] 계산된 관절 각도 (도):")
        print(f"q1: {ik_angles[0]}, q2: {ik_angles[1]}, q3: {ik_angles[2]}, q4: {ik_angles[3]}, q5: {ik_angles[4]}")

        # 검증: IK로 찾은 해를 다시 FK에 넣어 원래 목표와 일치하는지 확인
        print("\n--- 역기구학(IK) 결과 검증 ---")
        final_pos, final_R = fk_5dof_shoulder_custom(*ik_angles)

        # 최종 자세를 RPY로 변환하여 각도 확인
        final_rpy = Rotation.from_matrix(final_R).as_euler('zyx', degrees=True)  # [요, 피치, 롤]

        pos_error = np.linalg.norm(np.array(pos_xyz) - final_pos)
        pitch_error = abs(pitch_deg - final_rpy[1])

        print(f"최종 위치: {np.round(final_pos, 4)}")
        print(f"최종 자세(요,피치,롤): {np.round(final_rpy, 2)}°")
        print(f"최종 위치 오차: {pos_error:.6f} m")
        print(f"최종 피치 오차: {pitch_error:.6f}°")
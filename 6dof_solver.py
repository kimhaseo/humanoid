import numpy as np
import time


class RealTimeArmController:
    def __init__(self):
        # 링크 길이 설정
        self.L_base_sh = 0.1
        self.A1, self.A2 = 0.05, 0.05
        self.L_upper, self.L_lower, self.L_end = 0.3, 0.3, 0.1
        self.q = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def dh_matrix(self, alpha, a, d, theta):
        """DH 변환 행렬 생성"""
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
            [0, 0, 0, 1]
        ])

    def get_full_pose(self, q):
        """[중요] 관절 각도 q로부터 전체 변환 행렬(4x4)을 구함"""
        t1 = self.dh_matrix(0, 0, self.L_base_sh, q[0])
        t2 = self.dh_matrix(np.pi / 2, self.A1, 0, q[1])
        t3 = self.dh_matrix(np.pi / 2, self.A2, self.L_upper, q[2])
        t4 = self.dh_matrix(0, self.L_lower, 0, q[3])
        t5 = self.dh_matrix(0, 0, 0, q[4])
        t6 = self.dh_matrix(np.pi / 2, 0, self.L_end, q[5])
        return t1 @ t2 @ t3 @ t4 @ t5 @ t6

    def rotation_matrix_to_euler(self, R):
        """회전 행렬을 오일러 각도(deg)로 변환 (Roll, Pitch, Yaw)"""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        return np.degrees([x, y, z])

    def update(self, target_pos, gain=0.2):
        # 현재 위치만 추출하여 에러 계산
        T_curr = self.get_full_pose(self.q)
        current_pos = T_curr[:3, 3]
        error = target_pos - current_pos

        # 수치적 자코비안 생성
        epsilon = 1e-6
        J = np.zeros((3, 6))
        for i in range(6):
            q_plus = self.q.copy()
            q_plus[i] += epsilon
            J[:, i] = (self.get_full_pose(q_plus)[:3, 3] - current_pos) / epsilon

        # Damped Least Squares 역행렬
        lambd = 0.01
        JJT = J @ J.T
        J_inv = J.T @ np.linalg.inv(JJT + lambd ** 2 * np.eye(3))

        delta_q = J_inv @ error
        self.q += delta_q * gain
        return self.q, np.linalg.norm(error)


if __name__ == "__main__":
    arm = RealTimeArmController()
    target = np.array([0.2, 0.15, 0.4])

    print(f"목표 지점 {target}으로 수렴 시작...")

    start_total = time.perf_counter()
    count = 0

    while True:
        count += 1
        new_q, error_m = arm.update(target, gain=0.2)
        if error_m < 0.0001:  # 0.1mm 기준
            break

    end_total = time.perf_counter()

    # 최종 상태 추출
    final_T = arm.get_full_pose(new_q)
    final_pos = final_T[:3, 3]  # 최종 위치 (x, y, z)
    final_ori = arm.rotation_matrix_to_euler(final_T[:3, :3])  # 최종 방향 (RPY)

    print("-" * 50)
    print(f"총 수렴 시간: {(end_total - start_total) * 1000:.2f} ms")
    print(f"총 반복 횟수: {count} 회")
    print("-" * 50)
    print(f"1. 도달 위치 (x, y, z) [m]:\n   {np.round(final_pos, 4)}")
    print(f"2. 도달 포즈 (Roll, Pitch, Yaw) [deg]:\n   {np.round(final_ori, 2)}")
    print(f"3. 최종 위치 오차 [mm]: {error_m * 1000:.4f}")
    print(f"4. 최종 관절 각도 [deg]:\n   {np.round(np.degrees(new_q), 2)}")
    print("-" * 50)
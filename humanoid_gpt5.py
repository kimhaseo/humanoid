import numpy as np

def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])

def fk_5dof_shoulder_custom(q1_deg, q2_deg, q3_deg, q4_deg, q5_deg):
    q1 = np.radians(q1_deg)  # 관절1 y축 회전
    q2 = np.radians(q2_deg)  # 관절2 x축 회전
    q3 = np.radians(q3_deg)  # 관절3 z축 회전
    q4 = np.radians(q4_deg)  # 관절4 y축 회전
    q5 = np.radians(q5_deg)  # 관절5 y축 회전

    R1 = rotation_y(q1)
    R2 = rotation_x(q2)
    R3 = rotation_z(q3)
    R4 = rotation_y(q4)
    R5 = rotation_y(q5)

    offset_joint2 = np.array([0, 0.05, 0])
    pos_joint2 = R1 @ offset_joint2

    link2_vector = np.array([0, 0, -0.06])
    pos_joint3 = pos_joint2 + R1 @ R2 @ link2_vector

    link3_vector = np.array([0, 0, -0.06])
    pos_joint4 = pos_joint3 + R1 @ R2 @ R3 @ link3_vector

    link4_vector = np.array([0.07, 0, 0])
    pos_joint5 = pos_joint4 + R1 @ R2 @ R3 @ R4 @ link4_vector

    link5_vector = np.array([0.05, 0, 0])  # x축 방향 50mm
    end_pos = pos_joint5 + R1 @ R2 @ R3 @ R4 @ R5 @ link5_vector

    R_end = R1 @ R2 @ R3 @ R4 @ R5

    return end_pos, R_end

# 테스트
pos, R = fk_5dof_shoulder_custom(0, 0, 0, 0, 0)

pos_rounded = np.round(pos, 4)
R_rounded = np.round(R, 4)

print("엔드포인트 위치 (m):", pos_rounded)
print("엔드포인트 회전 행렬:\n", R_rounded)

import pybullet as p
import pybullet_data
import time

# ---------------------------
# PyBullet 초기화
# ---------------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)
timestep = 1./960.
p.setTimeStep(timestep)

# ---------------------------
# 로봇 및 위치
# ---------------------------
left_pos = [-0.5, 0, 0]
right_pos = [0.5, 0, 0]
left_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=left_pos)
right_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=right_pos)
ee_link_index = 6  # 엔드이펙터

# ---------------------------
# 공 생성 (충돌 없음)
# ---------------------------
sphere_radius = 0.05
vis_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1,0,0,1])
sphere_pos = [0,0,0.5]
sphere_id = p.createMultiBody(baseMass=0,
                              baseCollisionShapeIndex=-1,
                              baseVisualShapeIndex=vis_sphere,
                              basePosition=sphere_pos)

# ---------------------------
# 시뮬레이션 루프
# ---------------------------
steps_per_second = 240
for step in range(4000):
    # 몇 초 경과했는지 계산
    phase = (step // steps_per_second) % 2  # 0=왼쪽 차례, 1=오른쪽 차례
    step_in_phase = step % steps_per_second

    # 목표 위치 계산 (0~0.5초 공, 0.5~1초 원위치)
    if step_in_phase < steps_per_second // 2:
        # 왼쪽 로봇이 공, 오른쪽 로봇 원위치
        if phase == 0:
            left_target = sphere_pos
            right_target = [right_pos[0], right_pos[1], 0.5]
        else:
            right_target = sphere_pos
            left_target = [left_pos[0], left_pos[1], 0.5]
    else:
        # 공 터치 후 바로 원위치로 돌아오기
        left_target = [left_pos[0], left_pos[1], 0.5]
        right_target = [right_pos[0], right_pos[1], 0.5]

    # IK 계산
    left_joint_positions = p.calculateInverseKinematics(left_id, ee_link_index, left_target)
    right_joint_positions = p.calculateInverseKinematics(right_id, ee_link_index, right_target)

    # 관절 적용
    for j in range(p.getNumJoints(left_id)):
        info = p.getJointInfo(left_id, j)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.setJointMotorControl2(left_id, j, p.POSITION_CONTROL, left_joint_positions[j], maxVelocity=10)
    for j in range(p.getNumJoints(right_id)):
        info = p.getJointInfo(right_id, j)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.setJointMotorControl2(right_id, j, p.POSITION_CONTROL, right_joint_positions[j], maxVelocity=10)

    # 시뮬레이션 스텝
    p.stepSimulation()
    time.sleep(timestep)

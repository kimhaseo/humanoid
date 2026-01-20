import pybullet as p
import pybullet_data
import numpy as np
import time

# ---------------------------
# 초기화
# ---------------------------
physicsClient = p.connect(p.GUI)  # GUI 모드
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)
timestep = 1./240.
p.setTimeStep(timestep)

# ---------------------------
# 로봇 및 위치
# ---------------------------
left_pos = [-0.5, 0, 0]
right_pos = [0.5, 0, 0]
left_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=left_pos)
right_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=right_pos)
ee_link_index = 6

# ---------------------------
# 공 생성 (물리 적용)
# ---------------------------
sphere_radius = 0.05
mass = 0.05
col_sphere = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
vis_sphere = p.createVisualShape(p.GEOM_SPHERE, radius=sphere_radius, rgbaColor=[1,0,0,1])
sphere_id = p.createMultiBody(baseMass=mass,
                              baseCollisionShapeIndex=col_sphere,
                              baseVisualShapeIndex=vis_sphere,
                              basePosition=[0,0,0.5])
# 물리 속성
p.changeDynamics(sphere_id, -1, restitution=0.9, lateralFriction=0.1)

# ---------------------------
# 카메라 설정 함수
# ---------------------------
cam_width, cam_height = 64, 64
fov, near, far = 60, 0.01, 2

def capture_ee_camera(robot_id, ee_index):
    state = p.getLinkState(robot_id, ee_index)
    pos = state[0]
    orn = state[1]
    rot_matrix = p.getMatrixFromQuaternion(orn)
    cam_vec = [rot_matrix[2], rot_matrix[5], rot_matrix[8]]  # forward
    up_vec = [rot_matrix[1], rot_matrix[4], rot_matrix[7]]
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=pos,
        cameraTargetPosition=[pos[0]+cam_vec[0], pos[1]+cam_vec[1], pos[2]+cam_vec[2]],
        cameraUpVector=up_vec
    )
    proj_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=1, nearVal=near, farVal=far)
    img = p.getCameraImage(cam_width, cam_height, viewMatrix=view_matrix, projectionMatrix=proj_matrix)
    rgb_image = np.array(img[2])[:, :, :3]  # RGB
    depth_image = np.array(img[3])  # Depth
    return rgb_image, depth_image

# ---------------------------
# 강화학습용 최소 시뮬레이션 루프
# ---------------------------
steps_per_switch = 240  # 1초 주기
left_target_default = [left_pos[0], left_pos[1], 0.5]
right_target_default = [right_pos[0], right_pos[1], 0.5]

for step in range(4000):
    step_in_phase = step % steps_per_switch
    phase = (step // steps_per_switch) % 2  # 0=왼쪽 차례, 1=오른쪽 차례

    # ---------------------------
    # 공 목표 위치 결정
    # ---------------------------
    if step_in_phase < steps_per_switch // 2:
        # 차례 로봇이 공 위치로 이동
        if phase == 0:
            left_target = p.getBasePositionAndOrientation(sphere_id)[0]
            right_target = right_target_default
        else:
            right_target = p.getBasePositionAndOrientation(sphere_id)[0]
            left_target = left_target_default
    else:
        # 공 터치 후 원위치로 돌아오기
        left_target = left_target_default
        right_target = right_target_default

    # ---------------------------
    # IK 계산
    # ---------------------------
    left_joint_positions = p.calculateInverseKinematics(left_id, ee_link_index, left_target)
    right_joint_positions = p.calculateInverseKinematics(right_id, ee_link_index, right_target)

    # ---------------------------
    # 관절 적용 (부드럽게 이동)
    # ---------------------------
    for j in range(p.getNumJoints(left_id)):
        info = p.getJointInfo(left_id, j)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.setJointMotorControl2(left_id, j, p.POSITION_CONTROL,
                                    targetPosition=left_joint_positions[j], maxVelocity=1.5)

    for j in range(p.getNumJoints(right_id)):
        info = p.getJointInfo(right_id, j)
        if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.setJointMotorControl2(right_id, j, p.POSITION_CONTROL,
                                    targetPosition=right_joint_positions[j], maxVelocity=1.5)

    # ---------------------------
    # 카메라 관찰
    # ---------------------------
    if step % 10 == 0:  # 너무 자주 캡처하면 느려짐
        left_rgb, left_depth = capture_ee_camera(left_id, ee_link_index)
        right_rgb, right_depth = capture_ee_camera(right_id, ee_link_index)
        # 여기서 CNN 입력용으로 RGB+Depth 사용 가능

    # ---------------------------
    # 시뮬레이션 스텝
    # ---------------------------
    p.stepSimulation()
    time.sleep(timestep)

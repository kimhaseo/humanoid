from dataclasses import dataclass
import numpy as np

@dataclass
class Link:
    axis: np.ndarray      # rotation axis in local coordinates (3,)
    offset: np.ndarray    # translation after this joint in local coordinates (3,)

def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def rodrigues(axis, theta):
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K @ K)

def homogenous(R, t):
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = t
    return T

class SerialChain:
    def __init__(self, links):
        self.links = links
        self.n = len(links)

    def forward(self, thetas):
        T = np.eye(4)
        Ts = []  # transform after rotation of joint i (base->joint_i frame)
        for i, link in enumerate(self.links):
            R = rodrigues(link.axis, thetas[i])
            T = T @ homogenous(R, np.zeros(3))   # rotate
            Ts.append(T.copy())
            T = T @ homogenous(np.eye(3), link.offset)  # translate along offset
        return T, Ts

    def jacobian_pos(self, thetas):
        T_end, Ts = self.forward(thetas)
        p_end = T_end[:3,3]
        J = np.zeros((3, self.n))
        for i in range(self.n):
            R_i = Ts[i][:3,:3]
            axis_world = R_i @ self.links[i].axis
            p_i = Ts[i][:3,3]
            J[:,i] = np.cross(axis_world, (p_end - p_i))
        return J

    def jacobian_rz(self, thetas):
        """
        Jacobian mapping joint velocities to derivative of end-effector z-axis (3 x n)
        Using: d(r_z)/dt = sum_j (omega_j x r_z) where omega_j = axis_world * theta_dot_j
        So column j = axis_world x r_z
        """
        T_end, Ts = self.forward(thetas)
        r_z = T_end[:3,2]  # end-effector z-axis in world frame (3,)
        Jr = np.zeros((3, self.n))
        for i in range(self.n):
            R_i = Ts[i][:3,:3]
            axis_world = R_i @ self.links[i].axis
            Jr[:,i] = np.cross(axis_world, r_z)
        return Jr, r_z

    def inverse_kinematics_with_horizontal_ee(self, target_pos, thetas0=None,
                                              max_iters=300, tol_pos=1e-3, tol_rot=1e-3,
                                              alpha=0.8, lam=1e-2, w_rot=1.0):
        """
        Solve IK for position + constraint that end-effector z-axis == world z-axis ([0,0,1])
        - target_pos: (3,)
        - w_rot: weight for rotational constraint (relative importance)
        Returns: thetas, success(boolean), history_positions
        """
        if thetas0 is None:
            thetas = np.zeros(self.n)
        else:
            thetas = thetas0.copy()

        desired_rz = np.array([0.0, 0.0, 1.0])
        history = []

        for it in range(max_iters):
            T_end, _ = self.forward(thetas)
            p = T_end[:3,3]
            err_pos = target_pos - p  # 3

            Jr, r_z = self.jacobian_rz(thetas)  # Jr: 3 x n
            # rotational error: we want r_z -> desired_rz
            # use vector difference (could also use cross), then take x,y components (roll/pitch)
            err_r_vec = desired_rz - r_z   # 3
            err_rot = err_r_vec[:2]  # only x,y components (we don't constrain yaw)
            # Corresponding Jacobian rows: take the first two rows of Jr (effect on r_z.x and r_z.y)
            Jr_xy = Jr[:2, :]  # 2 x n

            # Build combined task: [pos(3); w_rot * rot(2)]
            J_pos = self.jacobian_pos(thetas)  # 3 x n
            J_comb = np.vstack((J_pos, w_rot * Jr_xy))  # (5 x n)
            err_comb = np.concatenate((err_pos, w_rot * err_rot))  # (5,)

            err_norm = np.linalg.norm(err_pos) + np.linalg.norm(err_rot)
            history.append(p.copy())
            if (np.linalg.norm(err_pos) < tol_pos) and (np.linalg.norm(err_rot) < tol_rot):
                return thetas, True, np.array(history)

            # Damped least squares for combined task:
            A = J_comb @ J_comb.T + (lam * np.eye(J_comb.shape[0]))
            # Solve A * v = err_comb  (v: task-space delta)
            try:
                delta_task = np.linalg.solve(A, err_comb)
            except np.linalg.LinAlgError:
                delta_task = np.linalg.lstsq(A, err_comb, rcond=None)[0]
            delta_theta = J_comb.T @ delta_task

            thetas += alpha * delta_theta
            thetas = (thetas + np.pi) % (2*np.pi) - np.pi

            # adapt damping (heuristic)
            if np.linalg.norm(delta_theta) > 1.0:
                lam *= 10
            elif np.linalg.norm(delta_theta) < 1e-4:
                lam = max(1e-6, lam / 2)

        return thetas, False, np.array(history)

# --- Build the robot from your description (same as before) ---
mm = 1.0
links = [
    Link(axis=np.array([0,1,0]), offset=np.array([0, 50*mm, 0])),
    Link(axis=np.array([1,0,0]), offset=np.array([0, 0, 30*mm])),
    Link(axis=np.array([0,0,1]), offset=np.array([0, 0, 60*mm])),
    Link(axis=np.array([0,1,0]), offset=np.array([0, 0, 40*mm])),
    Link(axis=np.array([0,1,0]), offset=np.array([0, 0, 30*mm])),
]

robot = SerialChain(links)

# --- Usage example ---
if __name__ == "__main__":
    target = np.array([80.0, 50.0, 120.0])  # 현실적인 목표 (mm)
    thetas_init = np.deg2rad(np.array([0, 10, 0, 0, 0]))
    thetas, success, hist = robot.inverse_kinematics_with_horizontal_ee(
        target_pos=target,
        thetas0=thetas_init,
        max_iters=800,
        tol_pos=1e-2,
        tol_rot=1e-3,
        alpha=0.9,
        lam=1e-2,
        w_rot=2.0  # 자세 제약의 중요도를 늘리려면 키우세요
    )
    print("Success:", success)
    print("Joint angles (deg):", np.rad2deg(thetas))
    T_end, _ = robot.forward(thetas)
    print("End-effector position (mm):", T_end[:3,3])
    print("End-effector z-axis (world):", T_end[:3,2])
    print("Target (mm):", target)

from dataclasses import dataclass
import numpy as np
from math import atan2, acos, sqrt, sin, cos, pi


# -------------------------- Robot Definition --------------------------
@dataclass
class Link:
    axis: np.ndarray  # rotation axis in local coordinates (3,)
    offset: np.ndarray  # translation after this joint in local coordinates (3,)


mm = 1.0
links = [
    Link(axis=np.array([0, 1, 0]), offset=np.array([0, 50 * mm, 0])),  # 어깨 Y축
    Link(axis=np.array([1, 0, 0]), offset=np.array([0, 0, 30 * mm])),  # 팔 위쪽 X축
    Link(axis=np.array([0, 0, 1]), offset=np.array([0, 0, 60 * mm])),  # 팔 Z축
    Link(axis=np.array([0, 1, 0]), offset=np.array([0, 0, 40 * mm])),  # 팔 아래 Y축
    Link(axis=np.array([0, 1, 0]), offset=np.array([0, 0, 30 * mm])),  # 손 끝 Y축
]


# -------------------------- Math Helpers --------------------------
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def rodrigues(axis, theta):
    a = axis / np.linalg.norm(axis)
    K = skew(a)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def homogenous(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def wrap(a):
    return (a + pi) % (2 * pi) - pi


# -------------------------- Serial Chain --------------------------
class SerialChain:
    def __init__(self, links):
        self.links = links
        self.n = len(links)

    def forward(self, thetas):
        T = np.eye(4)
        Ts = []
        for i, L in enumerate(self.links):
            R = rodrigues(L.axis, thetas[i])
            T = T @ homogenous(R, np.zeros(3))
            Ts.append(T.copy())
            T = T @ homogenous(np.eye(3), L.offset)
        return T, Ts

    def jacobian_pos(self, thetas):
        T_end, Ts = self.forward(thetas)
        p = T_end[:3, 3]
        J = np.zeros((3, self.n))
        for i in range(self.n):
            R_i = Ts[i][:3, :3]
            axis_w = R_i @ self.links[i].axis
            o_i = Ts[i][:3, 3]
            J[:, i] = np.cross(axis_w, p - o_i)
        return J

    def jacobian_r(self, thetas):
        T_end, Ts = self.forward(thetas)
        r_z = T_end[:3, 2]
        Jr = np.zeros((3, self.n))
        for i in range(self.n):
            R_i = Ts[i][:3, :3]
            axis_w = R_i @ self.links[i].axis
            Jr[:, i] = np.cross(axis_w, r_z)
        return Jr, r_z


# -------------------------- DLS Refine with Relaxed Z --------------------------
def refine_with_relaxed_z(chain: SerialChain, target, q_init,
                          max_iters=200, tol_pos=1e-3, tol_rot=1e-3,
                          alpha=0.9, lam=1e-2, w_rot=5.0):
    q = np.array(q_init, dtype=float)
    history = []
    for it in range(max_iters):
        T_end, _ = chain.forward(q)
        p = T_end[:3, 3]
        err_pos = target - p

        Jr, r_z = chain.jacobian_r(q)
        err_rot = -r_z[:2]  # x/y components → pitch/roll 맞추기
        Jr_xy = Jr[:2, :]

        Jp = chain.jacobian_pos(q)
        J_comb = np.vstack((Jp, w_rot * Jr_xy))
        err_comb = np.concatenate((err_pos, w_rot * err_rot))

        history.append((p.copy(), r_z.copy()))

        if np.linalg.norm(err_pos) < tol_pos and np.linalg.norm(err_rot) < tol_rot:
            return q, True, history

        A = J_comb @ J_comb.T + lam * np.eye(J_comb.shape[0])
        try:
            v = np.linalg.solve(A, err_comb)
        except np.linalg.LinAlgError:
            v = np.linalg.lstsq(A, err_comb, rcond=None)[0]
        delta_q = J_comb.T @ v
        q += alpha * delta_q
        q = np.vectorize(wrap)(q)

    return q, False, history


# -------------------------- IK Pipeline --------------------------
def ik_relaxed_z_pipeline(target, q_init=None):
    chain = SerialChain(links)

    # 초기값 지정: 사용자가 넣으면 그걸 쓰고, 없으면 0으로
    if q_init is None:
        q0 = np.zeros(len(links))
    else:
        q0 = np.array(q_init)

    q_final, success, hist = refine_with_relaxed_z(chain, target, q0,
                                                   max_iters=300, tol_pos=1e-3, tol_rot=1e-3,
                                                   alpha=0.9, lam=1e-2, w_rot=5.0)
    T_end, _ = chain.forward(q_final)
    return {
        "q_init_deg": np.degrees(q0),
        "q_final_deg": np.degrees(q_final),
        "success": success,
        "ee_pos": T_end[:3, 3],
        "ee_rz": T_end[:3, 2],
        "history": hist
    }


# -------------------------- Usage Example --------------------------
if __name__ == "__main__":
    target = np.array([80.0, 50.0, 90.0])

    # 초기값 직접 지정 가능 (예: 다 0도)
    q_init = [0, 0, 0, 90, 0]  # 5관절 초기값
    out = ik_relaxed_z_pipeline(target, q_init=q_init)

    print("init (deg):", out["q_init_deg"])
    print("final (deg):", out["q_final_deg"])
    print("success:", out["success"])
    print("EE pos (mm):", out["ee_pos"])
    print("EE z (world):", out["ee_rz"])

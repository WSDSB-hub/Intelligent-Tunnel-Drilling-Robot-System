# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==================== 机械臂真实参数 ====================
BASE_HEIGHT = 2.7
L1 = 5.028
L2 = 4.359
D5_MIN, D5_MAX = 0.0, 3.5

THETA1_RANGE = [0, 80]
THETA2_RANGE = [0, 80]
THETA3_RANGE = [-45, 45]
THETA4_RANGE = [0, 360]

# ★ 最优台车距离
TRUCK_DISTANCE = 9.0
# 原始距离（钻孔点坐标对应的参考距离）
REFERENCE_DISTANCE = 11.5
# =======================================================

def dh_transform(theta, d, a, alpha):
    theta_rad = np.deg2rad(theta)
    alpha_rad = np.deg2rad(alpha)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,      ca,      d],
        [0,         0,       0,      1]
    ])

def forward_kinematics(theta1, theta2, theta3, theta4, d5):
    T1 = dh_transform(theta1, BASE_HEIGHT, 0, 90)
    T2 = dh_transform(theta2, 0, L1, 0)
    T3 = dh_transform(theta3, 0, 0, 90)
    T4 = dh_transform(theta4, 0, L2, 0)
    T5 = dh_transform(0, d5, 0, 0)
    T = T1 @ T2 @ T3 @ T4 @ T5
    return T[:3, 3]

def get_arm_points(theta1, theta2, theta3, theta4, d5):
    T1 = dh_transform(theta1, BASE_HEIGHT, 0, 90)
    T2 = dh_transform(theta2, 0, L1, 0)
    T3 = dh_transform(theta3, 0, 0, 90)
    T4 = dh_transform(theta4, 0, L2, 0)
    T5 = dh_transform(0, d5, 0, 0)
    p0 = np.array([0, 0, 0, 1])
    p1 = T1 @ p0
    p2 = T1 @ T2 @ p0
    p3 = T1 @ T2 @ T3 @ p0
    p4 = T1 @ T2 @ T3 @ T4 @ p0
    p5 = T1 @ T2 @ T3 @ T4 @ T5 @ p0
    return np.array([p0[:3], p1[:3], p2[:3], p3[:3], p4[:3], p5[:3]])

def check_reachability(target_point):
    for t1 in np.arange(THETA1_RANGE[0], THETA1_RANGE[1]+1, 15):
        for t2 in np.arange(THETA2_RANGE[0], THETA2_RANGE[1]+1, 15):
            for t3 in np.arange(THETA3_RANGE[0], THETA3_RANGE[1]+1, 15):
                for t4 in np.arange(THETA4_RANGE[0], THETA4_RANGE[1]+1, 45):
                    for d5 in np.arange(D5_MIN, D5_MAX+0.1, 0.5):
                        pos = forward_kinematics(t1, t2, t3, t4, d5)
                        if np.linalg.norm(pos - target_point) < 0.3:
                            for t1_f in np.arange(max(t1-10, THETA1_RANGE[0]), min(t1+10, THETA1_RANGE[1])+1, 3):
                                for t2_f in np.arange(max(t2-10, THETA2_RANGE[0]), min(t2+10, THETA2_RANGE[1])+1, 3):
                                    for t3_f in np.arange(max(t3-10, THETA3_RANGE[0]), min(t3+10, THETA3_RANGE[1])+1, 3):
                                        for t4_f in np.arange(max(t4-20, THETA4_RANGE[0]), min(t4+20, THETA4_RANGE[1])+1, 10):
                                            for d5_f in np.arange(max(d5-0.3, D5_MIN), min(d5+0.3, D5_MAX)+0.01, 0.1):
                                                pos_f = forward_kinematics(t1_f, t2_f, t3_f, t4_f, d5_f)
                                                if np.linalg.norm(pos_f - target_point) < 0.15:
                                                    return True, (t1_f, t2_f, t3_f, t4_f, d5_f), np.linalg.norm(pos_f - target_point)
    best_dist = np.inf
    best_joints = None
    for t1 in np.arange(THETA1_RANGE[0], THETA1_RANGE[1]+1, 10):
        for t2 in np.arange(THETA2_RANGE[0], THETA2_RANGE[1]+1, 10):
            for t3 in np.arange(THETA3_RANGE[0], THETA3_RANGE[1]+1, 10):
                for t4 in np.arange(THETA4_RANGE[0], THETA4_RANGE[1]+1, 30):
                    for d5 in np.arange(D5_MIN, D5_MAX+0.01, 0.3):
                        pos = forward_kinematics(t1, t2, t3, t4, d5)
                        dist = np.linalg.norm(pos - target_point)
                        if dist < best_dist:
                            best_dist = dist
                            best_joints = (t1, t2, t3, t4, d5)
    return False, best_joints, best_dist

def load_face_pointcloud(filepath, max_points=8000):
    try:
        data = np.loadtxt(filepath)
        xyz = data[:, :3] / 1000.0
        pts = xyz
        mask = (pts[:, 0] > 5.0) & (pts[:, 0] < 18.0) & (pts[:, 1] > -1.5) & (pts[:, 1] < 3.5)
        xyz = xyz[mask]
        if len(xyz) > max_points:
            idx = np.random.choice(len(xyz), max_points, replace=False)
            xyz = xyz[idx]
        return xyz
    except:
        return None

def main():
    print("=" * 60)
    print(f"凿岩台车机械臂可达性分析 (最优距离 = {TRUCK_DISTANCE}m)")
    print("=" * 60)

    print("\n加载掌子面点云...")
    face_pts = load_face_pointcloud("20260707_170626_origin.txt")
    if face_pts is not None:
        print(f"掌子面点数: {len(face_pts)}")
    else:
        print("未找到点云文件")

    try:
        drill_points = np.loadtxt("real_drill_plan.txt", skiprows=1)
        if drill_points.ndim == 1:
            drill_points = drill_points.reshape(1, -1)
        print(f"钻孔点数: {len(drill_points)}")
    except:
        print("未找到 real_drill_plan.txt")
        return

    # ★ 平移钻孔点坐标（模拟台车移动到最优距离）
    adjusted_points = drill_points.copy()
    adjusted_points[:, 0] += (TRUCK_DISTANCE - REFERENCE_DISTANCE)

    print("\n验证可达性（两阶段搜索）...")
    reachable = []
    best_joints_list = []
    for i, pt in enumerate(adjusted_points):
        ok, joints, dist = check_reachability(pt)
        reachable.append(ok)
        best_joints_list.append(joints)
        status = "✓" if ok else f"✗ (最近{dist:.2f}m)"
        print(f"  点{i+1:3d}: ({pt[0]:5.2f}, {pt[1]:5.2f}, {pt[2]:5.2f}) -> {status}")

    num_ok = sum(reachable)
    print(f"\n可达性结果: {num_ok}/{len(adjusted_points)} ({num_ok/len(adjusted_points)*100:.1f}%)")

    # 可视化
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')
    if face_pts is not None:
        ax.scatter(face_pts[:, 0], face_pts[:, 1], face_pts[:, 2],
                   c='darkgray', s=0.8, alpha=0.4, label='Tunnel face')
    ax.scatter([0], [0], [0], c='black', s=200, marker='s', label='Truck base')
    ax.scatter([0], [0], [BASE_HEIGHT], c='brown', s=150, marker='^', label='Arm base')

    ws = []
    for _ in range(2000):
        t1 = np.random.uniform(*THETA1_RANGE)
        t2 = np.random.uniform(*THETA2_RANGE)
        t3 = np.random.uniform(*THETA3_RANGE)
        t4 = np.random.uniform(*THETA4_RANGE)
        d5 = np.random.uniform(D5_MIN, D5_MAX)
        ws.append(forward_kinematics(t1, t2, t3, t4, d5))
    ws = np.array(ws)
    ax.scatter(ws[:, 0], ws[:, 1], ws[:, 2],
               c='lightsteelblue', s=1, alpha=0.08, label='Workspace envelope')

    pts = np.array(adjusted_points)
    if sum(~np.array(reachable)) > 0:
        ax.scatter(pts[~np.array(reachable), 0], pts[~np.array(reachable), 1], pts[~np.array(reachable), 2],
                   c='red', marker='x', s=40, linewidth=1.5, label=f'Unreachable ({sum(~np.array(reachable))})')
    if num_ok > 0:
        ax.scatter(pts[np.array(reachable), 0], pts[np.array(reachable), 1], pts[np.array(reachable), 2],
                   c='limegreen', s=120, edgecolors='black', linewidth=1.5, label=f'Reachable ({num_ok})')

    arm_colors = ['blue', 'darkcyan', 'teal', 'navy', 'steelblue']
    arm_labeled = False
    for i, (pt, ok, joints) in enumerate(zip(adjusted_points, reachable, best_joints_list)):
        if ok and joints is not None:
            arm_pts = get_arm_points(*joints)
            c = arm_colors[i % len(arm_colors)]
            lbl = 'Arm to reachable pt' if not arm_labeled else None
            ax.plot(arm_pts[:, 0], arm_pts[:, 1], arm_pts[:, 2],
                    '-o', color=c, linewidth=2.5, markersize=5, label=lbl)
            arm_labeled = True

    ref_pts = get_arm_points(45, 45, 0, 0, 2.0)
    ax.plot(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2],
            '--', color='gray', linewidth=1.5, alpha=0.5, label='Reference pose')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Reachability (Truck at {TRUCK_DISTANCE}m) | {num_ok}/{len(adjusted_points)} reachable')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.savefig("reachability_final.png", dpi=200)
    print("\n最终可达性图已保存为 reachability_final.png")
    plt.show()

if __name__ == "__main__":
    main()
    
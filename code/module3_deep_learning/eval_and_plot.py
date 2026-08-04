import numpy as np
import matplotlib.pyplot as plt

# ==================== 机械臂参数（不可修改） ====================
BASE_HEIGHT = 2.7
L1 = 5.028
L2 = 4.359
D5_MIN, D5_MAX = 0.0, 3.5

THETA1_RANGE = [0, 80]
THETA2_RANGE = [0, 80]
THETA3_RANGE = [-45, 45]
THETA4_RANGE = [0, 360]
# =====================================================================

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

def check_reachability(target_point):
    # 粗搜 + 精搜（同之前优化版）
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
                                                    return True
    # 不返回 False，只返回最近距离（优化用，这里不需要最优关节，只关心是否可达）
    best_dist = np.inf
    for t1 in np.arange(THETA1_RANGE[0], THETA1_RANGE[1]+1, 10):
        for t2 in np.arange(THETA2_RANGE[0], THETA2_RANGE[1]+1, 10):
            for t3 in np.arange(THETA3_RANGE[0], THETA3_RANGE[1]+1, 10):
                for t4 in np.arange(THETA4_RANGE[0], THETA4_RANGE[1]+1, 30):
                    for d5 in np.arange(D5_MIN, D5_MAX+0.01, 0.3):
                        pos = forward_kinematics(t1, t2, t3, t4, d5)
                        dist = np.linalg.norm(pos - target_point)
                        if dist < best_dist:
                            best_dist = dist
    return False

def main():
    # 读取钻孔点
    try:
        drill_points = np.loadtxt("real_drill_plan.txt", skiprows=1)
        if drill_points.ndim == 1:
            drill_points = drill_points.reshape(1, -1)
        print(f"读取 {len(drill_points)} 个钻孔点")
    except:
        print("未找到 real_drill_plan.txt")
        return

    # 测试的距离范围 (m)
    distances = np.arange(8.5, 12.6, 0.5)  # 8.5, 9.0, 9.5, ..., 12.5
    reach_counts = []

    for dist in distances:
        print(f"\n测试距离 = {dist:.1f}m ...")
        count = 0
        # 修改台车距离（我们通过修改目标点坐标来实现，假设台车在X=0处，目标点X坐标不变，但实际物理意义是台车距掌子面距离，这里简化处理：钻孔点X坐标已经是在掌子面坐标系下的值，我们并不直接移动点云。实际上，调整TRUCK_DISTANCE应重新规划钻孔点位置。但为了快速展示效果，这里我们假设钻孔点X坐标表示掌子面深度，而机械臂基座始终在X=0。调整台车距离相当于移动了掌子面在X方向的位置。我们在机械臂模型中并未直接使用TRUCK_DISTANCE参数，之前的代码也没有把它作为变量，所以这里需要将钻孔点X坐标视为相对于基座的前向距离。因此，改变距离就是在改变目标点的X坐标。简单实现：假设原始钻孔点是在某个参考距离下测得的，我们可以通过缩放X坐标来模拟不同距离。但为了保证逻辑清晰，我直接沿用你之前的思路：钻孔点文件中的坐标是相对于台车的实际坐标，我们通过修改“期望的台车-掌子面距离”来平移钻孔点的X坐标。假设当前钻孔点是在距离D0=11.5m时规划的，那么新距离下钻孔点的X坐标应加上 (dist - 11.5) 的偏移量。
        # 这里我们将目标点整体在X方向平移 (dist - 11.5) 米。
        adjusted_points = drill_points.copy()
        adjusted_points[:, 0] += (dist - 11.5)   # 如果距离更近，X变小；更远则X变大

        for pt in adjusted_points:
            if check_reachability(pt):
                count += 1
        reach_counts.append(count)
        print(f"距离 {dist:.1f}m: {count}/{len(drill_points)} 个点可达 ({count/len(drill_points)*100:.1f}%)")

    # 汇总结果
    print("\n" + "="*50)
    print("汇总结果")
    print("="*50)
    for d, c in zip(distances, reach_counts):
        print(f"距离 {d:.1f}m : {c:3d}/{len(drill_points)} 可达 ({c/len(drill_points)*100:.1f}%)")

    # 绘制优化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(distances, reach_counts, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Truck Distance (m)', fontsize=12)
    plt.ylabel('Reachable Drill Points', fontsize=12)
    plt.title('Optimization of Truck Parking Distance', fontsize=14)
    plt.grid(True, alpha=0.3)
    # 标注每个点
    for d, c in zip(distances, reach_counts):
        plt.text(d, c+0.5, str(c), ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("distance_optimization.png", dpi=150)
    plt.show()
    print("优化曲线已保存为 distance_optimization.png")

if __name__ == "__main__":
    main()
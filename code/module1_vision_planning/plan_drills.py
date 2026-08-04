import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ========== 1. 重新生成带坑洼的点云（与上一步相同） ==========
print("生成虚拟掌子面...")
width, height = 4.0, 5.0
resolution = 0.02
x = np.arange(-width/2, width/2, resolution)
y = np.arange(-height/2, height/2, resolution)
X, Y = np.meshgrid(x, y)
Z = 0.2 * np.sin(X * 1.5) + 0.1 * np.cos(Y * 1.2)

def add_hole(X, Y, Z, cx, cy, r, depth):
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist < r
    Z[mask] -= depth * (np.cos(np.pi * dist[mask] / r) * 0.5 + 0.5)
    return Z

Z = add_hole(X, Y, Z, -1.0, 0.8, 0.4, 0.15)
Z = add_hole(X, Y, Z, 1.2, -0.5, 0.5, 0.2)
Z = add_hole(X, Y, Z, 0.2, 1.8, 0.35, 0.12)
Z += np.random.normal(scale=0.005, size=Z.shape)

# 转为Open3D点云
points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.8, 0.8, 0.8])

# ========== 2. 坑洼检测（复用上一步逻辑） ==========
print("检测坑洼...")
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
pts = np.asarray(pcd.points)
curvature = np.zeros(len(pts))
for i in range(len(pts)):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], 0.06)
    if k < 4:
        curvature[i] = 0.0
        continue
    neighbors = pts[idx, :]
    cov = np.cov(neighbors.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    curvature[i] = eigenvalues[0]
threshold = np.percentile(curvature, 90)
hole_mask = curvature > threshold

# 标记坑洼为红色
colors = np.asarray(pcd.colors)
colors[hole_mask] = [1, 0, 0]
pcd.colors = o3d.utility.Vector3dVector(colors)

# ========== 3. 钻孔点位规划 ==========
print("正在规划钻孔点位...")
# 设定爆破参数（可调整）
drill_spacing_x = 0.4   # 孔距（米）
drill_spacing_y = 0.5   # 排距（米）
margin = 0.3            # 边界留边距离

# 生成规则的候选钻孔网格（投影到X-Y平面）
x_centers = np.arange(-width/2 + margin, width/2 - margin, drill_spacing_x)
y_centers = np.arange(-height/2 + margin, height/2 - margin, drill_spacing_y)
cand_x, cand_y = np.meshgrid(x_centers, y_centers)
cand_x = cand_x.flatten()
cand_y = cand_y.flatten()

# 为每个候选点找到它在点云中最接近的实际Z坐标（投影到表面）
def find_surface_z(x, y, pts, search_radius=0.05):
    # 找XY平面内距离最近的点的Z值
    dist_xy = np.sqrt((pts[:, 0] - x)**2 + (pts[:, 1] - y)**2)
    min_idx = np.argmin(dist_xy)
    if dist_xy[min_idx] < search_radius:
        return pts[min_idx, 2]
    else:
        return None

valid_drills = []
for cx, cy in zip(cand_x, cand_y):
    z_surface = find_surface_z(cx, cy, pts)
    if z_surface is None:
        continue
    # 检查该候选点是否太靠近坑洼区域
    # 简单判据：该点本身是否在坑洼掩膜内（取投影最近的点）
    dist_all = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)
    nearest_idx = np.argmin(dist_all)
    if hole_mask[nearest_idx]:
        continue  # 落在坑洼上，跳过
    valid_drills.append([cx, cy, z_surface])

valid_drills = np.array(valid_drills)
print(f"共规划 {len(valid_drills)} 个有效钻孔点，剔除 {len(cand_x) - len(valid_drills)} 个无效点")

# ========== 4. 可视化：在原有点云上叠加钻孔点 ==========
# 创建钻孔点球体（红色小球）
drill_spheres = []
for drill in valid_drills:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.04)
    sphere.translate(drill)
    sphere.paint_uniform_color([0, 1, 0])  # 绿色钻孔点
    drill_spheres.append(sphere)

# 显示带坑洼的点云和绿色钻孔点
o3d.visualization.draw_geometries(
    [pcd] + drill_spheres,
    window_name="Drilling Plan - 绿色=钻孔点，红色=坑洼"
)

# ========== 5. 输出钻孔点坐标到文本文件 ==========
np.savetxt("drill_plan.txt", valid_drills, header="x y z", comments='', fmt='%.4f')
print("钻孔点坐标已保存到 drill_plan.txt")
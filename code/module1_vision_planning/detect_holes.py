import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# ========== 1. 生成带坑洼的虚拟掌子面点云 ==========
print("正在生成虚拟掌子面点云...")
width, height = 4.0, 5.0  # 断面宽高（米）
resolution = 0.02         # 点间距
x = np.arange(-width/2, width/2, resolution)
y = np.arange(-height/2, height/2, resolution)
X, Y = np.meshgrid(x, y)

# 略带弯曲的岩壁表面
Z = 0.2 * np.sin(X * 1.5) + 0.1 * np.cos(Y * 1.2)

# 制造坑洼的函数
def add_hole(X, Y, Z, center_x, center_y, radius, depth):
    dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = dist < radius
    Z[mask] -= depth * (np.cos(np.pi * dist[mask] / radius) * 0.5 + 0.5)
    return Z

# 添加三个坑洼
Z = add_hole(X, Y, Z, -1.0, 0.8, 0.4, 0.15)
Z = add_hole(X, Y, Z, 1.2, -0.5, 0.5, 0.2)
Z = add_hole(X, Y, Z, 0.2, 1.8, 0.35, 0.12)

# 加一点噪声模拟岩石粗糙表面
Z += np.random.normal(scale=0.005, size=Z.shape)

# 转为 Open3D 点云
points = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=1)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 初始灰色

print(f"点云已生成，包含 {len(pcd.points)} 个点")

# ========== 2. 计算曲率并检测坑洼 ==========
print("正在计算表面曲率并检测坑洼...")
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
    curvature[i] = eigenvalues[0]  # 最小特征值作为曲率指标

# 取曲率前10%的点标记为坑洼
threshold = np.percentile(curvature, 90)
hole_mask = curvature > threshold

# 坑洼点染成红色
colors = np.asarray(pcd.colors)
colors[hole_mask] = [1, 0, 0]
pcd.colors = o3d.utility.Vector3dVector(colors)

print(f"检测完成，阈值={threshold:.6f}，坑洼点数量={np.sum(hole_mask)}")

# ========== 3. 显示点云 ==========
print("正在打开三维点云窗口（可用鼠标旋转/缩放）...")
o3d.visualization.draw_geometries([pcd], window_name="Hole Detection - 坑洼检测结果")

# ========== 4. 显示曲率分布直方图 ==========
plt.figure()
plt.hist(curvature, bins=50, alpha=0.7, color='steelblue')
plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.4f}')
plt.xlabel("Curvature (平面拟合残差)")
plt.ylabel("Point Count")
plt.title("Surface Curvature Distribution")
plt.legend()
plt.show()

print("实验完成！")
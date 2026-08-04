import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

# ======================== 参数配置区 ========================
# 点云文件路径
POINT_CLOUD_FILE = "20260707_170626_origin.txt"

# 降采样体素大小（米），数值越大点数越少，运行越快
VOXEL_SIZE = 0.03

# 掌子面裁剪范围（根据实际数据调整）
Y_MIN, Y_MAX = -1.5, 3.5   # 掌子面高度方向
X_MIN, X_MAX = 5.0, 18.0   # 掌子面宽度方向

# 坑洼检测参数
SEARCH_RADIUS = 0.05       # 邻域搜索半径（米），根据点密度调整
HOLE_PERCENTILE = 90       # 曲率分位数阈值，越大坑洼点越少

# 钻孔规划参数（需根据爆破设计网格图修改）
DRILL_SPACING_X = 0.5      # 孔距（米）
DRILL_SPACING_Y = 0.6      # 排距（米）
MARGIN = 0.4               # 边界留边（米）
MAX_XY_DIST = 0.5          # 钻孔候选点与最近点云的最大允许XY距离

# 是否启用坑洼避让（True=避开红色区域，False=不避开）
AVOID_HOLES = False        # 先设为False看整体分布，满意后再改为True
# ============================================================

# ========== 1. 加载点云 ==========
print("加载真实点云...")
data = np.loadtxt(POINT_CLOUD_FILE)
xyz = data[:, :3] / 1000.0   # 毫米 -> 米

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
print(f"原始点数：{len(pcd.points)}")

# 降采样
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
print(f"降采样后点数：{len(pcd.points)}")

# 裁剪掌子面核心区域
print("裁剪掌子面核心区域...")
pts_np = np.asarray(pcd.points)
mask = (pts_np[:, 0] > X_MIN) & (pts_np[:, 0] < X_MAX) & \
       (pts_np[:, 1] > Y_MIN) & (pts_np[:, 1] < Y_MAX)
pcd = pcd.select_by_index(np.where(mask)[0])
print(f"裁剪后掌子面点数：{len(pcd.points)}")
pcd.paint_uniform_color([0.8, 0.8, 0.8])

# 在裁剪完成后，坑洼检测之前添加
o3d.io.write_point_cloud("cropped_face.ply", pcd)
print("裁剪后点云已保存为 cropped_face.ply")

# ========== 2. 坑洼检测 ==========
print("检测坑洼...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=SEARCH_RADIUS, max_nn=30)
)

pcd_tree = o3d.geometry.KDTreeFlann(pcd)
pts = np.asarray(pcd.points)
curvature = np.zeros(len(pts))

for i in range(len(pts)):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(
        pcd.points[i], SEARCH_RADIUS * 1.2
    )
    if k < 4:
        curvature[i] = 0.0
        continue
    neighbors = pts[idx, :]
    cov = np.cov(neighbors.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    curvature[i] = eigenvalues[0]  # 最小特征值表示平面拟合残差

threshold = np.percentile(curvature, HOLE_PERCENTILE)
hole_mask = curvature > threshold

colors = np.asarray(pcd.colors)
colors[hole_mask] = [1, 0, 0]   # 坑洼标记为红色
pcd.colors = o3d.utility.Vector3dVector(colors)
print(f"坑洼点数：{np.sum(hole_mask)}")

# ========== 3. 钻孔规划 ==========
print("规划钻孔...")
pts_array = np.asarray(pcd.points)

# 使用裁剪后的实际边界
x_min, x_max = pts_array[:, 0].min(), pts_array[:, 0].max()
y_min, y_max = pts_array[:, 1].min(), pts_array[:, 1].max()

x_centers = np.arange(x_min + MARGIN, x_max - MARGIN, DRILL_SPACING_X)
y_centers = np.arange(y_min + MARGIN, y_max - MARGIN, DRILL_SPACING_Y)
cand_x, cand_y = np.meshgrid(x_centers, y_centers)
cand_x = cand_x.flatten()
cand_y = cand_y.flatten()
print(f"候选钻孔点数：{len(cand_x)}")

# 构建二维 XY 平面的 KD 树（仅用前两列）
xy_pts = pts_array[:, :2]
xy_tree = cKDTree(xy_pts)

valid_drills = []
for cx, cy in zip(cand_x, cand_y):
    dist, idx = xy_tree.query([cx, cy])
    if dist > MAX_XY_DIST:
        continue
    # 坑洼避让
    if AVOID_HOLES and hole_mask[idx]:
        continue
    z_surface = pts_array[idx, 2]
    valid_drills.append([cx, cy, z_surface])

valid_drills = np.array(valid_drills)
print(f"有效钻孔点数：{len(valid_drills)}")

# ========== 4. 可视化与保存 ==========
# 平移点云到几何中心，便于观察
center = pcd.get_center()
pcd_translated = pcd.translate(-center)

drill_spheres = []
for drill in valid_drills:
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
    sphere.translate(drill - center)
    sphere.paint_uniform_color([0, 1, 0])   # 绿色钻孔点
    drill_spheres.append(sphere)

# 保存原始坐标的钻孔点（不包含平移）
np.savetxt("real_drill_plan.txt", valid_drills,
           header="x y z", fmt='%.4f', comments='')
print("钻孔点坐标已保存到 real_drill_plan.txt")

# 弹出三维窗口
o3d.visualization.draw_geometries(
    [pcd_translated] + drill_spheres,
    window_name="Real Face Drilling Plan (绿色=钻孔点，红色=坑洼)"
)
print("窗口已弹出。")
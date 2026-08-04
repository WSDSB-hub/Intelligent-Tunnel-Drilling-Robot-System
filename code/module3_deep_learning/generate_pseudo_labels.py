import numpy as np
import open3d as o3d
import os

# ================= 参数 =================
INPUT_FILE = "20260707_170626_origin.txt"
OUTPUT_DIR = "pseudo_data"
NUM_SAMPLES = 20          # 生成多少个样本块
BLOCK_SIZE = 2.0          # 每个块的大小（米）
VOXEL_SIZE = 0.03         # 降采样
SEARCH_RADIUS = 0.05      # 曲率计算半径
HOLE_PERCENTILE = 75      # 坑洼阈值
# ========================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("加载原始点云...")
data = np.loadtxt(INPUT_FILE)
xyz = data[:, :3] / 1000.0   # 毫米转米

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# 裁剪掌子面核心区域
pts = np.asarray(pcd.points)
mask = (pts[:, 0] > 5.0) & (pts[:, 0] < 18.0) & (pts[:, 1] > -1.5) & (pts[:, 1] < 3.5)
pcd = pcd.select_by_index(np.where(mask)[0])
pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
pts_full = np.asarray(pcd.points)
print(f"裁剪后点数: {len(pts_full)}")

# 计算曲率
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=SEARCH_RADIUS, max_nn=30))
pcd_tree = o3d.geometry.KDTreeFlann(pcd)
curvature = np.zeros(len(pts_full))
for i in range(len(pts_full)):
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], SEARCH_RADIUS * 1.2)
    if k < 4:
        curvature[i] = 0.0
        continue
    neighbors = pts_full[idx, :]
    cov = np.cov(neighbors.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    curvature[i] = eigenvalues[0]
threshold_global = np.percentile(curvature, HOLE_PERCENTILE)

# 生成样本
print(f"生成 {NUM_SAMPLES} 个训练样本...")
for sample_id in range(NUM_SAMPLES):
    # 随机选择一个中心点
    center_idx = np.random.randint(0, len(pts_full))
    center = pts_full[center_idx, :2]  # XY平面中心

    # 截取 block
    dx = np.abs(pts_full[:, 0] - center[0])
    dy = np.abs(pts_full[:, 1] - center[1])
    block_mask = (dx < BLOCK_SIZE/2) & (dy < BLOCK_SIZE/2)
    if np.sum(block_mask) < 100:
        continue  # 点太少跳过

    block_pts = pts_full[block_mask]
    block_curv = curvature[block_mask]
    # 基于全局阈值或局部阈值均可，这里用全局阈值
    labels = (block_curv > threshold_global).astype(int)

    # 归一化坐标（可选，提升训练稳定性）
    block_pts_centered = block_pts - np.mean(block_pts, axis=0)
    scale = np.max(np.abs(block_pts_centered))
    if scale > 0:
        block_pts_centered /= scale

    # 保存为文件：每行 x y z label
    save_data = np.column_stack((block_pts_centered, labels))
    filename = os.path.join(OUTPUT_DIR, f"sample_{sample_id:04d}.txt")
    np.savetxt(filename, save_data, fmt="%.6f", header="x y z label", comments='')
    print(f"  样本 {sample_id}: {len(block_pts)} 个点 -> {filename}")

print("伪标签数据生成完成。")
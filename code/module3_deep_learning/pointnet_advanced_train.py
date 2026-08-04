import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ================= 参数 =================
DATA_DIR = "pseudo_data"
BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 0.001
NUM_POINTS = 2048              # 增加到2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ========================================

# ---------- T-Net 空间变换网络 ----------
class TNet(nn.Module):
    """用于学习3D空间变换矩阵"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        # 初始化fc3的权重为0，偏置为单位矩阵
        nn.init.zeros_(self.fc3.weight)
        nn.init.eye_(self.fc3.bias.view(k, k))

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]  # 全局池化
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        # 初始化为单位矩阵附近
        x = x.view(-1, self.k, self.k)
        return x

# ---------- 升级版 PointNet 分割模型 ----------
class PointNetSegAdvanced(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.input_transform = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # 分割头
        self.conv5 = nn.Conv1d(512 + 64 + 3, 256, 1)
        self.conv6 = nn.Conv1d(256, 128, 1)
        self.conv7 = nn.Conv1d(128, num_classes, 1)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        # x: (B, 3, N)
        # 输入变换
        trans = self.input_transform(x)
        x_trans = torch.bmm(trans, x)  # (B, 3, N)

        x1 = F.relu(self.bn1(self.conv1(x_trans)))     # (B, 64, N)
        x2 = F.relu(self.bn2(self.conv2(x1)))          # (B, 128, N)
        x3 = F.relu(self.bn3(self.conv3(x2)))          # (B, 256, N)
        x4 = self.bn4(self.conv4(x3))                  # (B, 512, N)
        global_feat = torch.max(x4, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, x.shape[2])

        combined = torch.cat((x1, x_trans, global_feat), dim=1)
        out = F.relu(self.bn5(self.conv5(combined)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.conv7(out)
        return out

# ---------- 带数据增强的数据集 ----------
class TunnelDatasetAug(Dataset):
    def __init__(self, data_dir, num_points, files=None, augment=False):
        if files is None:
            self.files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
        else:
            self.files = files
        self.data_dir = data_dir
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = os.path.join(self.data_dir, self.files[idx])
        data = np.loadtxt(filepath, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        points = data[:, :3].astype(np.float32)
        labels = data[:, 3].astype(np.int64)

        # 数据增强
        if self.augment:
            # 随机旋转（绕Z轴）
            theta = np.random.uniform(0, 2*np.pi)
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta),  np.cos(theta), 0],
                                [0, 0, 1]], dtype=np.float32)
            points = points @ rot_mat.T
            # 随机平移
            points += np.random.normal(0, 0.02, size=points.shape).astype(np.float32)
            # 随机加噪
            points += np.random.normal(0, 0.005, size=points.shape).astype(np.float32)

        # 采样/填充
        if len(points) >= self.num_points:
            choice = np.random.choice(len(points), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_points, replace=True)
        points = points[choice]
        labels = labels[choice]

        points = torch.from_numpy(points).transpose(0, 1)  # (3, N)
        labels = torch.from_numpy(labels)                  # (N,)
        return points, labels

# ---------- 计算 IoU ----------
def compute_iou(pred, target, num_classes=2):
    """计算每个类别的IoU和平均IoU"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

# ---------- 主程序 ----------
print("加载数据并划分训练/验证集...")
all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_dataset = TunnelDatasetAug(DATA_DIR, NUM_POINTS, files=train_files, augment=True)
val_dataset = TunnelDatasetAug(DATA_DIR, NUM_POINTS, files=val_files, augment=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

# 模型、损失、优化器
model = PointNetSegAdvanced(num_classes=2).to(DEVICE)
# 类别加权：假设坑洼点(label=1)占比约10%，设置较高权重
weights = torch.tensor([0.3, 0.7]).to(DEVICE)  # 正常点权重0.3，坑洼点权重0.7
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print("开始训练...")
train_losses, val_losses = [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    # 训练
    model.train()
    total_train_loss = 0.0
    for points, labels in train_loader:
        points, labels = points.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(points)
        outputs = outputs.permute(0, 2, 1).contiguous().view(-1, 2)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证
    model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for points, labels in val_loader:
            points, labels = points.to(DEVICE), labels.to(DEVICE)
            outputs = model(points)
            outputs_flat = outputs.permute(0, 2, 1).contiguous().view(-1, 2)
            loss = criterion(outputs_flat, labels.view(-1))
            total_val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()

    # 计算验证集IoU
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        all_preds_cat = torch.cat(all_preds)
        all_labels_cat = torch.cat(all_labels)
        ious = compute_iou(all_preds_cat, all_labels_cat)
        iou_str = " ".join([f"Cls{c}: {iou:.3f}" for c, iou in enumerate(ious)])
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU: {iou_str}")
    else:
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "pointnet_advanced_best.pth")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("loss_curve.png", dpi=150)
print("损失曲线已保存为 loss_curve.png")
plt.show()

# 最终IoU报告
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for points, labels in val_loader:
        points, labels = points.to(DEVICE), labels.to(DEVICE)
        outputs = model(points)
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
all_preds_cat = torch.cat(all_preds)
all_labels_cat = torch.cat(all_labels)
ious = compute_iou(all_preds_cat, all_labels_cat)
print("\n最终验证集IoU:")
print(f"  正常岩石 (Class 0): {ious[0]:.4f}")
print(f"  坑洼区域 (Class 1): {ious[1]:.4f}")
print(f"  平均IoU: {np.nanmean(ious):.4f}")

# 可视化一个验证样本
val_iter = iter(val_loader)
points, labels = next(val_iter)
points = points[:1].to(DEVICE)
with torch.no_grad():
    pred = model(points)
    pred_labels = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
labels = labels[0].cpu().numpy()
points_np = points[0].transpose(0,1).cpu().numpy()

fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(points_np[:,0], points_np[:,1], points_np[:,2], c=labels, cmap='coolwarm', s=3)
ax1.set_title('Ground Truth')
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(points_np[:,0], points_np[:,1], points_np[:,2], c=pred_labels, cmap='coolwarm', s=3)
ax2.set_title('Prediction (Advanced)')
ax3 = fig.add_subplot(133)
ax3.bar(['Normal Rock', 'Hole/Fracture'], [ious[0], ious[1]], color=['blue', 'red'])
ax3.set_ylabel('IoU')
ax3.set_title('Per-Class IoU')
plt.tight_layout()
plt.savefig("pointnet_advanced_result.png", dpi=150)
plt.show()
print("高级模型结果已保存为 pointnet_advanced_result.png")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

# ==================== 可调参数 ====================
MWD_FILE = "mwd_data.mwd"
OUTPUT_IMAGE = "drilling_states_advanced.png"

SMOOTH_WINDOW = 15
SEGMENT_SIZE = 5

# 阈值（根据诊断数据校准）
FP_LOW = 20.0
DP_LOW = 20.0
HP_HARD = 180.0          # 上调硬岩门槛（原100→180）
DP_HARD = 65.0           # 硬岩需 DP 也偏高
HP_NORMAL_LOW = 100.0    # 正常钻进 HP 下限
RP_HIGH = 55.0
PR_LOW = 0.15
WF_NORMAL_LOW = 30.0

DP_RISE_RATIO = 1.5
FP_RISE_RATIO = 1.5
# =================================================

def time_str_to_seconds(t):
    try:
        parts = t.split(':')
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            return h * 3600 + m * 60 + s
    except:
        pass
    return 0.0

def load_mwd(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if '[MWD DATA]' in line:
            data_start = i + 1
            break
    header = lines[data_start].strip().split('\t')
    data_rows = []
    for line in lines[data_start+1:]:
        line = line.strip()
        if not line or line.startswith('['):
            break
        parts = line.split('\t')
        if len(parts) == len(header):
            converted = []
            for j, val in enumerate(parts):
                col = header[j]
                if 'time' in col.lower():
                    converted.append(val)
                else:
                    try:
                        converted.append(float(val))
                    except ValueError:
                        converted.append(np.nan)
            data_rows.append(converted)
    df = pd.DataFrame(data_rows, columns=header)
    time_sec = df['Time'].apply(time_str_to_seconds).values
    time_sec -= time_sec[0]
    return df, time_sec

def extract_columns(df):
    def get_col(keyword):
        cols = [c for c in df.columns if keyword in c]
        if cols:
            return pd.to_numeric(df[cols[0]], errors='coerce').values
        return np.zeros(len(df))
    return {
        'FP': get_col('FP'),
        'DP': get_col('DP'),
        'HP': get_col('HP'),
        'RP': get_col('RP'),
        'PR': get_col('PR'),
        'WF': get_col('WF'),
        'WP': get_col('WP')
    }

def smooth_signals(signals, window_length):
    smoothed = {}
    for name, sig in signals.items():
        if len(sig) > window_length:
            smoothed[name] = savgol_filter(sig, window_length, polyorder=3)
        else:
            smoothed[name] = sig.copy()
    return smoothed

def segment_stats(signals, time_sec, segment_size):
    n = len(time_sec)
    seg_indices = []
    seg_time = []
    seg_features = []
    for start in range(0, n, segment_size):
        end = min(start + segment_size, n)
        seg_indices.append((start, end))
        seg_time.append(time_sec[start])
        feats = {}
        for name, sig in signals.items():
            seg = sig[start:end]
            feats[name + '_mean'] = np.nanmean(seg)
            feats[name + '_std'] = np.nanstd(seg)
            feats[name + '_max'] = np.nanmax(seg)
            feats[name + '_min'] = np.nanmin(seg)
            feats[name + '_range'] = np.ptp(seg) if len(seg) > 1 else 0.0
        seg_features.append(feats)
    return seg_indices, seg_time, seg_features

def classify_segment(feats, prev_feats=None):
    fp_mean = feats['FP_mean']
    dp_mean = feats['DP_mean']
    hp_mean = feats['HP_mean']
    rp_mean = feats['RP_mean']
    pr_mean = feats['PR_mean']
    wf_mean = feats['WF_mean']
    dp_std = feats['DP_std']

    # 1. 未接触 / 空打
    if (fp_mean < FP_LOW and dp_mean < DP_LOW) or (hp_mean < 10.0 and fp_mean < 30.0):
        return 'no_contact'

    # 2. 刚接触岩石（跃升检测）
    if prev_feats is not None:
        prev_fp = prev_feats['FP_mean']
        prev_dp = prev_feats['DP_mean']
        if (prev_fp < FP_LOW and prev_dp < DP_LOW) and \
           (fp_mean > prev_fp * FP_RISE_RATIO and dp_mean > prev_dp * DP_RISE_RATIO):
            return 'rock_contact'

    # 3. 卡钻风险
    if rp_mean > RP_HIGH and pr_mean < PR_LOW:
        return 'jamming_risk'

    # 4. 裂隙/破碎带：压力波动大
    if dp_std > 15.0:
        return 'fracture_zone'

    # 5. 硬岩：冲击压力非常高，且阻尼也高（两者同时偏高）
    if hp_mean > HP_HARD and dp_mean > DP_HARD:
        return 'hard_rock'

    # 6. 正常钻进：FP、DP、HP 都在正常范围，且水流正常
    if fp_mean > FP_LOW and dp_mean > DP_LOW and hp_mean > HP_NORMAL_LOW and wf_mean > WF_NORMAL_LOW:
        return 'normal_drilling'

    # 7. 兜底
    return 'transition'

def plot_advanced(time_sec, signals_smooth, signals_raw, seg_indices, seg_states, seg_time):
    state_colors = {
        'no_contact': 'lightgray',
        'rock_contact': 'gold',
        'normal_drilling': 'limegreen',
        'hard_rock': 'red',
        'fracture_zone': 'orange',
        'jamming_risk': 'darkred',
        'transition': 'cyan'
    }

    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    plot_configs = [
        ('FP', 'Feed Pressure (bar)', 'blue'),
        ('DP', 'Damping Pressure (bar)', 'red'),
        ('HP', 'Percussive Pressure (bar)', 'darkorange'),
        ('RP', 'Rotation Pressure (bar)', 'purple'),
        ('WF', 'Water Flow (l/min)', 'steelblue')
    ]

    for ax in axes:
        for idx, (start, end) in enumerate(seg_indices):
            state = seg_states[idx]
            color = state_colors.get(state, 'white')
            t_start = time_sec[start]
            t_end = time_sec[end-1] if end > start else t_start
            ax.axvspan(t_start, t_end, alpha=0.15, color=color)

    for idx, (start, end) in enumerate(seg_indices):
        state = seg_states[idx]
        color = state_colors.get(state, 'white')
        t_start = time_sec[start]
        if idx % 3 == 0:
            axes[0].text(t_start, axes[0].get_ylim()[1]*0.9, state[:8],
                         fontsize=7, color=color, rotation=45)

    for ax, (name, ylabel, color) in zip(axes, plot_configs):
        if name in signals_raw:
            raw = signals_raw[name]
            smoothed = signals_smooth[name]
            t = time_sec
            ax.plot(t, raw, alpha=0.25, color=color, linewidth=0.8)
            ax.plot(t, smoothed, color=color, linewidth=1.8, label=name)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (seconds)')
    axes[0].set_title('Advanced Drilling State Classification (tuned)')

    legend_patches = [Patch(facecolor=state_colors[s], alpha=0.4,
                            label=s.replace('_',' ').title())
                      for s in ['no_contact', 'rock_contact', 'normal_drilling',
                                 'hard_rock', 'fracture_zone', 'jamming_risk']]
    axes[0].legend(handles=legend_patches, loc='upper left', fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=150)
    print(f"高级触觉感知图已保存为：{OUTPUT_IMAGE}")
    plt.show()

def main():
    print("加载 MWD 数据...")
    df, time_sec = load_mwd(MWD_FILE)
    print(f"数据点数：{len(df)}")

    print("提取关键信号...")
    signals_raw = extract_columns(df)

    print("平滑信号...")
    signals_smooth = smooth_signals(signals_raw, SMOOTH_WINDOW)

    print(f"分段统计（窗口大小={SEGMENT_SIZE}）...")
    seg_indices, seg_time, seg_features = segment_stats(signals_smooth, time_sec, SEGMENT_SIZE)

    print("对每段进行分类...")
    seg_states = []
    for i, feats in enumerate(seg_features):
        prev = seg_features[i-1] if i > 0 else None
        state = classify_segment(feats, prev)
        seg_states.append(state)

    print("\n=== 钻进状态统计（分段） ===")
    unique, counts = np.unique(seg_states, return_counts=True)
    total = len(seg_states)
    for s, c in zip(unique, counts):
        print(f"{s:20s}: {c:3d} 段 ({c/total*100:.1f}%)")

    print("\n=== 诊断：每段 FP/DP/HP 均值 ===")
    for i, feats in enumerate(seg_features):
        state = seg_states[i]
        fp_m = feats['FP_mean']
        dp_m = feats['DP_mean']
        hp_m = feats['HP_mean']
        rp_m = feats['RP_mean']
        pr_m = feats['PR_mean']
        print(f"段{i:2d} | FP={fp_m:6.1f} DP={dp_m:6.1f} HP={hp_m:6.1f} RP={rp_m:5.1f} PR={pr_m:5.2f} | {state}")

    print("绘制高级分析图...")
    plot_advanced(time_sec, signals_smooth, signals_raw, seg_indices, seg_states, seg_time)

if __name__ == "__main__":
    main()
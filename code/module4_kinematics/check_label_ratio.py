import numpy as np
import os

total = 0
holes = 0
for f in os.listdir('pseudo_data'):
    if f.endswith('.txt'):
        data = np.loadtxt(f'pseudo_data/{f}', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        labels = data[:, 3].astype(int)
        total += len(labels)
        holes += np.sum(labels)

print(f'总点数: {total}')
print(f'坑洼点: {holes} ({holes/total*100:.1f}%)')
print(f'正常点: {total-holes} ({(total-holes)/total*100:.1f}%)')
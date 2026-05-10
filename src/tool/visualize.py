import numpy as np
#使用../../out/train1文件夹下的- epoch_losses.npy epoch_accuracies.npy 
import matplotlib.pyplot as plt

# 加载数据
epoch_losses = np.load('../../out/train/deeper/epoch_losses.npy')
epoch_accuracies = np.load('../../out/train/deeper/epoch_accuracies.npy')

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 绘制loss曲线
ax1.plot(epoch_losses, 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Curve', fontsize=14)
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# 绘制accuracy曲线
ax2.plot(epoch_accuracies, 'r-', linewidth=2, label='Training Accuracy')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Training Accuracy Curve', fontsize=14)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../../out/train/deeper/training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse, Polygon

x = [1,3,5,7,9,11,13]  # 横坐标类别
data1 = [85.21,86.12,85.67,87.04,86.58,85.52,84.91]           # 第一组数据
data2 = [46.06,46.64,47.23,47.67,46.79,47.08,46.35]           # 第二组数据

fig, axs = plt.subplots(1, 2, figsize=(10, 5), 
                        gridspec_kw={'width_ratios': [1, 1]})
# 2. 计算位置与宽度
bar_width = 1.5                    # 单组柱宽（小于0.5）

bars1a = axs[0].bar(x , data1, width=bar_width, 
        color='#1f77b4', edgecolor='black', label='组1',hatch=['**'])
# bars1b = axs[0].bar(x + bar_width/2, data2, width=bar_width, 
#         color='#ff7f0e', edgecolor='black', label='组2',hatch=['\\\\','**'])

# 4. 设置坐标轴标签
axs[0].set_xticks(x)
axs[0].set_ylim(83, 87.5)  # 优化Y轴范围
axs[0].grid(axis='y',  visible=False)

# bars2a = axs[1].bar(x - bar_width/2, data3, width=bar_width, 
#         color='#1f77b4', edgecolor='black', label='组1',hatch=['o', 'O'])
bars2b = axs[1].bar(x , data2, width=bar_width, 
        color='#ff7f0e', edgecolor='black', label='组2',hatch=['OO'])

# 4. 设置坐标轴标签
axs[1].set_xticks(x)
axs[1].set_ylim(44, 48)  # 优化Y轴范围
axs[1].grid(axis='y',  visible=False)

# 6. 添加图例与显示
plt.tight_layout()  # 自动调整间距
plt.show()
plt.savefig('./show/distrubution.png', dpi=300)
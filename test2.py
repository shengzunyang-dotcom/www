# import torch.nn as nn
import torch
import pickle as pkl
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


with open('/data/yangshengzun/work2/checkpoint/proxies.pkl', 'rb') as f:
    proxies = pkl.load(f)
proxies = proxies.cpu()
mu = proxies[:,:384]
sigma = torch.nn.functional.softplus(proxies[:,384:])
samples_per_dist = 1000
all_samples = []
labels = []

for i in range(7):
    samples = np.random.normal(mu[i],sigma[i],size = (samples_per_dist,384))
    all_samples.append(samples)
    labels.extend([i] * samples_per_dist)

# 合并所有样本和标签
X = np.vstack(all_samples)  # 形状: (7 * 500, dim)
y = np.array(labels)        # 形状: (3500,)

tsne = TSNE(
    n_components=2,
    perplexity=30,          # 7个分布的理想值
    learning_rate=300,
    n_iter=1500,
    init='pca',
    random_state=42
)

# 执行降维
X_tsne = tsne.fit_transform(X)
palette = sns.color_palette("husl", 7)  # 7种鲜明颜色
for i in range(7):
    plt.figure(figsize=(10, 8))
    idx = (y == i)
    plt.scatter(
        X_tsne[idx, 0], X_tsne[idx, 1],
        color=palette[i], label=f'Dist {i}', alpha=0.6
    )
    sns.kdeplot(
        x=X_tsne[idx, 0], y=X_tsne[idx, 1],
        color=palette[i], alpha=0.3, thresh=0.1  # 调整thresh控制轮廓范围
    )
    plt.legend(title='Distribution ID')
    plt.title('t-SNE Visualization of 7 Distributions', fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(visible=False)
    plt.savefig(f'./show/class{i}.png', dpi=300)
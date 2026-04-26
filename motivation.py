import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取音频
audio_path = '/data/zhaoxianbing/multimodal-dataset/cmu/CMU-RAW/MOSI/Raw/Audio/WAV_16000/Segmented/_dI--eQ6qVU_23.wav'  # 替换成你的音频路径
y, sr = librosa.load(audio_path, sr=None)

# 稀疏帧参数
frame_length = int(0.025 * sr)
hop_length = int(0.01 * sr)

energy = np.array([
    np.sum(y[i:i+frame_length]**2)
    for i in range(0, len(y)-frame_length, hop_length)
])
mean_energy = np.mean(energy)

# 柱状图参数
indices = np.arange(len(energy))
bar_width = 0.6  # 小于1会有缝隙

fig, axs = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [2, 1]})

# 波形
librosa.display.waveshow(y, sr=sr, color='steelblue', ax=axs[0])
axs[0].set_title('Waveform')
axs[0].set_ylabel('Amplitude')
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)

# 能量分两类bar绘制
above = energy >= mean_energy
below = ~above

# 朝上：高于均值（黑色）
axs[1].bar(indices[above], energy[above]-mean_energy, bottom=mean_energy, width=bar_width, color='k', align='center', label='Above Mean')
# 朝下：低于均值（灰色）
axs[1].bar(indices[below], energy[below]-mean_energy, bottom=mean_energy, width=bar_width, color='gray', align='center', label='Below Mean')

axs[1].axhline(mean_energy, color='dodgerblue', linestyle='--', linewidth=1)
axs[1].set_title('Short-Time Energy')
axs[1].set_xlabel('Frame Index')
axs[1].set_ylabel('Energy')
axs[1].spines['top'].set_visible(False)
axs[1].spines['right'].set_visible(False)
axs[1].legend(frameon=False, loc='upper right')

plt.tight_layout()
# plt.show()
plt.savefig('./show/motivation.png')
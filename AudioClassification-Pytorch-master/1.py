import torchaudio.transforms as T
import torchaudio
import matplotlib.pyplot as plt

# 读取音频文件
waveform, sample_rate = torchaudio.load('C:\\Users\\Administrator\\Desktop\\Code\\AudioClassification-Pytorch-master\\dataset\\shipEar\\Dredger\\80__04_10_12_adricristuy-1.wav')

# 计算 MelSpectrogram
mel_spectrogram = T.MelSpectrogram()(waveform)

# 可视化 MelSpectrogram
plt.figure(figsize=(10, 4))
plt.imshow(torch.log(mel_spectrogram[0]), cmap='viridis', aspect='auto', origin='lower')
plt.title('MelSpectrogram')
plt.xlabel('Time')
plt.ylabel('Mel Frequency Bin')
plt.show()

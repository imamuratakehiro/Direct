"""2次元特徴量を時間方向に平均、全結合層で1次元化。"""

import torch
import torch.nn as nn

from utils.func import progress_bar
from ..csn import ConditionalSimNet2d


# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass

class To1D640timefreq(nn.Module):
    def __init__(self, in_channel, to1d_mode) -> None:
        super().__init__()
        self.to1d_mode = to1d_mode
        if to1d_mode == "mean_linear":
            self.fc1 = nn.Linear(in_channel, 640)
        elif to1d_mode == "meanstd_linear":
            self.fc1 = nn.Linear(in_channel*2, 640)
        elif to1d_mode == "pool":
            self.fc1 = nn.Linear(640*2, 640)
        self.to(device)

    def forward(self, input):
        if self.to1d_mode == "mean_linear":
            out_mean = torch.mean(input, dim=3)
            out_2d = out_mean.view(out_mean.size()[0], -1)
        elif self.to1d_mode == "meanstd_linear":
            out_mean = torch.mean(input, dim=3)
            out_std = torch.std(input, dim=3)
            out_2d = torch.concat([out_mean, out_std], dim=2).view(out_mean.size()[0], -1) # [B, ch*F*2]
        elif self.to1d_mode == "pool":
            avgpool = nn.AvgPool2d(kernel_size=(input.shape[2], input.shape[3]))
            # グローバル平均プーリング
            out_mean = avgpool(input)
            maxpool = nn.MaxPool2d(kernel_size=(input.shape[2], input.shape[3]))
            # グローバル最大プーリング
            out_max = maxpool(input)
            out_2d = torch.squeeze(torch.concat([out_mean, out_max], dim=1))

        output = self.fc1(out_2d)
        return output

class To1D640(nn.Module):
    def __init__(self, to1d_mode, order, in_channel) -> None:
        super().__init__()
        if order == "timefreq":
            self.to1d = To1D640timefreq(in_channel=int(in_channel), to1d_mode=to1d_mode)
        elif order =="freqtime":
            pass
        elif order == "freq_emb_time":
            pass
            #self.embnet = To1DFreqEmbedding(to1d_mode=to1d_mode, in_channel_freq=int(in_channel), tanh=tanh)
        elif order == "bilstm":
            pass
        #deviceを指定
        self.to(device)

    def forward(self, input):
        emb_vec = self.to1d(input)
        return emb_vec


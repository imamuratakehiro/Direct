# Created by zume06

import torch.nn as nn
import numpy as np
import torch

from ..csn import ConditionalSimNet2d, ConditionalSimNet1d
from ..to1d.model_embedding import EmbeddingNet128to128, To1dEmbedding
from ..to1d.model_linear import To1D640
from utils.func import normalize_torch, denormalize_torch, standardize_torch, destandardize_torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, middle_channels, kernel_size=3, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.rl = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            middle_channels, out_channels, kernel_size=3, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.rl(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, size):
        x = self.up(x, output_size = size)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class UNetEncoder(nn.Module):
    def __init__(self, encoder_in_size) -> None:
        super().__init__()
        self.TCB1 = TwoConvBlock(encoder_in_size, 32, 32)
        self.TCB2 = TwoConvBlock(32, 64, 64)
        self.TCB3 = TwoConvBlock(64, 128, 128)
        self.TCB4 = TwoConvBlock(128, 256, 256)
        self.TCB5 = TwoConvBlock(256, 512, 512)
        self.maxpool = nn.MaxPool2d(5, stride=2, padding=2)
    
    def forward(self, x):
        x = self.TCB1(x)
        x1 = x.clone()
        x = self.maxpool(x)
        # print(x.shape)

        x = self.TCB2(x)
        x2 = x.clone()
        x = self.maxpool(x)
        # print(x.shape)

        x = self.TCB3(x)
        x3 = x.clone()
        x = self.maxpool(x)
        # print(x.shape)

        x = self.TCB4(x)
        x4 = x.clone()
        x = self.maxpool(x)
        # print(x.shape)

        x5 = self.TCB5(x)
        # print(x.shape)
        return x1, x2, x3, x4, x5

class UNetDecoder(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.deconv5_a = nn.ConvTranspose2d(512, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = nn.Sequential(
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv4_a = nn.ConvTranspose2d(256+256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = nn.Sequential(
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv3_a = nn.ConvTranspose2d(128+128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = nn.Sequential(
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv2_a = nn.ConvTranspose2d(64+64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = nn.Sequential(
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2))
        self.deconv1_a = nn.ConvTranspose2d(32+32, 1, kernel_size = (5, 5), stride=(2, 2), padding=2)
        #deviceを指定
        self.to(device)
    def forward(self, sep_feature, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv5_out = self.deconv5_a(sep_feature, output_size = conv4_out.size())
        deconv5_out = self.deconv5_b(deconv5_out)
        deconv4_out = self.deconv4_a(torch.cat([deconv5_out, conv4_out], 1), output_size = conv3_out.size())
        deconv4_out = self.deconv4_b(deconv4_out)
        deconv3_out = self.deconv3_a(torch.cat([deconv4_out, conv3_out], 1), output_size = conv2_out.size())
        deconv3_out = self.deconv3_b(deconv3_out)
        deconv2_out = self.deconv2_a(torch.cat([deconv3_out, conv2_out], 1), output_size = conv1_out.size())
        deconv2_out = self.deconv2_b(deconv2_out)
        deconv1_out = self.deconv1_a(torch.cat([deconv2_out, conv1_out], 1), output_size = input.size())
        output = torch.sigmoid(deconv1_out)
        return output

class TripletModelZume(nn.Module):
    def __init__(self, cfg, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        if mono:
            encoder_in_size = 1
        else:
            encoder_in_size = 2
        if cfg.complex:
            encoder_in_size *= 2
        #encoder_out_size = len(inst_list) * 128
        # Encoder
        self.encoder = UNetEncoder(encoder_in_size)
        # Decoder
        self.decoder = UNetDecoder()
        # To1d
        #self.to1d = To1D640(to1d_mode=to1d_mode, order=order, in_channel=(f_size/2/(2**6)+1)*encoder_out_size)
        if mel:
            in_channel = (n_mels//(2**4)+1)*512
        else:
            in_channel = (f_size/2/(2**4)+1)*512
        self.to1d = To1D640(to1d_mode=to1d_mode, order=order, in_channel=in_channel)
        #self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        #deviceを指定
        self.inst_list = inst_list

    def forward(self, input):
        if self.cfg.standardize:
            input, mean, std = standardize_torch(input)
        elif self.cfg.normalize:
            input, max, min = normalize_torch(input)
        # Encoder
        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out = self.encoder(input)
        #print(conv5_out.shape)
        # to1d
        output_emb = self.to1d(conv5_out)
        # 原点からのユークリッド距離にlogをかけてsigmoidしたものを無音有音の確率とする
        csn1d = ConditionalSimNet1d(); csn1d.to(output_emb.device)
        output_probability = {inst : torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=output_emb.device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        return output_emb, output_probability

"""Tripletのためのモデル。UNet部分は出力640次元を128次元で条件付けしてDecoderに入力。Decoderは5つ"""

import torch
import torch.nn as nn
from torchinfo import summary
import math

from ..csn import ConditionalSimNet2d, ConditionalSimNet1d
from ..to1d.model_linear import To1D640
from utils.func import normalize_torch, denormalize_torch, standardize_torch, destandardize_torch

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

class MyError(Exception):
    pass


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, last=False) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = (5, 5), stride=(2, 2), padding=2))
        #if not last:
        self.conv.add_module("bn", nn.BatchNorm2d(out_channels))
        self.conv.add_module("rl", nn.LeakyReLU(0.2))
    def forward(self, input):
        return self.conv(input)

class UNetEncoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size):
        super().__init__()
        # Encoder
        self.conv1 = Conv2d(encoder_in_size, 16)
        self.conv2 = Conv2d(16, 32)
        self.conv3 = Conv2d(32, 64)
        self.conv4 = Conv2d(64, 128)
        self.conv5 = Conv2d(128, 256)
        self.conv6 = Conv2d(256, encoder_out_size, last=True)
        #deviceを指定
        #self.to(device)
    def forward(self, input):
        # Encoder
        conv1_out = self.conv1(input)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        return conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out

class UNetDecoder(nn.Module):
    def __init__(self, encoder_in_size, encoder_out_size) -> None:
        super().__init__()
        self.deconv6_a = nn.ConvTranspose2d(encoder_out_size, 256, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv6_b = nn.Sequential(
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv5_a = nn.ConvTranspose2d(256+256, 128, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv5_b = nn.Sequential(
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv4_a = nn.ConvTranspose2d(128+128, 64, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv4_b = nn.Sequential(
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(0.2),
                                nn.Dropout2d(0.5))
        self.deconv3_a = nn.ConvTranspose2d(64+64, 32, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv3_b = nn.Sequential(
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2))
        self.deconv2_a = nn.ConvTranspose2d(32+32, 16, kernel_size = (5, 5), stride=(2, 2), padding=2)
        self.deconv2_b = nn.Sequential(
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.2))
        self.deconv1_a = nn.ConvTranspose2d(16+16, encoder_in_size, kernel_size = (5, 5), stride=(2, 2), padding=2)
        #deviceを指定
        self.to(device)
    def forward(self, sep_feature, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input):
        deconv6_out = self.deconv6_a(sep_feature, output_size = conv5_out.size())
        deconv6_out = self.deconv6_b(deconv6_out)
        deconv5_out = self.deconv5_a(torch.cat([deconv6_out, conv5_out], 1), output_size = conv4_out.size())
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

class UNetForTriplet_2d_de5_to1d640(nn.Module):
    def __init__(self, cfg, inst_list, f_size, mono=True, to1d_mode="mean_linear", order="timefreq", mel=False, n_mels=259):
        super().__init__()
        if mono:
            encoder_in_size = 1
        else:
            encoder_in_size = 2
        if cfg.complex:
            encoder_in_size *= 2
        encoder_out_size = len(inst_list) * 128
        # Encoder
        self.encoder = UNetEncoder(encoder_in_size, encoder_out_size)
        # Decoder
        for inst in inst_list:
            if inst == "drums":
                self.decoder_drums = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "bass":
                self.decoder_bass = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "piano":
                self.decoder_piano = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "guitar":
                self.decoder_guitar = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "vocals":
                self.decoder_vocals = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "residuals":
                self.decoder_residuals = UNetDecoder(encoder_in_size, encoder_out_size)
            elif inst == "other":
                self.decoder_other = UNetDecoder(encoder_in_size, encoder_out_size)
        # To1d
        if mel:
            in_channel = math.ceil(n_mels/(2**6))*encoder_out_size
        else:
            in_channel = (f_size/2/(2**6)+1)*encoder_out_size
        self.to1d = To1D640(to1d_mode=to1d_mode, order=order, in_channel=in_channel)
        self.sigmoid = nn.Sigmoid()
        #deviceを指定
        self.to(device)
        self.inst_list = inst_list
        self.cfg = cfg


    def forward(self, input):
        # Encoder
        if self.cfg.standardize:
            input, mean, std = standardize_torch(input)
        elif self.cfg.normalize:
            input, max, min = normalize_torch(input)

        conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out = self.encoder(input)

        # 特徴量を条件づけ
        size = conv6_out.shape
        #print(size)
        # インスタンスを生成していなかったら、生成する。
        csn = ConditionalSimNet2d(size, device) # csnのモデルを保存されないようにするために配列に入れる
        decoder = {}
        output_decoder = {}
        for inst in self.inst_list:
            if inst == "drums":
                decoder[inst] = self.decoder_drums
            elif inst == "bass":
                decoder[inst] = self.decoder_bass
            elif inst == "piano":
                decoder[inst] = self.decoder_piano
            elif inst == "guitar":
                decoder[inst] = self.decoder_guitar
            elif inst == "vocals":
                decoder[inst] = self.decoder_vocals
            elif inst == "residuals":
                decoder[inst] = self.decoder_residuals
            elif inst == "other":
                decoder[inst] = self.decoder_other

        for idx, inst in enumerate(self.inst_list):
            # decoder
            sep_feature_decoder = csn(conv6_out, torch.tensor([idx], device=device))  # 特徴量を条件づけ
            decoder_out = decoder[inst].forward(sep_feature_decoder, conv5_out, conv4_out, conv3_out, conv2_out, conv1_out, input)
            if self.cfg.standardize:
                output_decoder[inst] = destandardize_torch(input*decoder_out, mean, std) # マスクをかけてからdestandardize
            elif self.cfg.normalize:
                output_decoder[inst] = denormalize_torch(input*decoder_out, max, min) # マスクをかけてからdenormalize
            else:
                output_decoder[inst] = input*decoder_out
        # to1d
        output_emb = self.to1d(conv6_out)
        # 原点からのユークリッド距離にlogをかけてsigmoidしたものを無音有音の確率とする
        csn1d = ConditionalSimNet1d() # csnのモデルを保存されないようにするために配列に入れる
        csn1d.to(output_emb.device)
        output_probability = {inst : torch.log(torch.sqrt(torch.sum(csn1d(output_emb, torch.tensor([i], device=device))**2, dim=1))) for i,inst in enumerate(self.inst_list)} # logit
        return output_emb, output_probability, output_decoder
    
def main():
    # モデルを定義
    inst_list = ["drums", "bass", "piano", "guitar", "residuals"]
    model = UNetForTriplet_2d_de5_to1d640(inst_list=inst_list, f_size=1024)
    batchsize = 16
    summary(model=model,
            input_size=(batchsize, 1, 513, 259),
            col_names=["input_size", "output_size", "num_params", "mult_adds"],
            depth=4)
    
if "__main__" == __name__:
    main()
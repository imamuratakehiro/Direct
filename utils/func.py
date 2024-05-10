import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as Fa
import time
import librosa
import os
import museval
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# GPUが使用可能かどうか判定、使用可能なら使用する
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n=== Using {device}({__name__}). ===\n")

def trackname(no):
    if no in range(1, 10):
        track_name = "Track0000{}".format(no)
    elif no in range(10, 100):
        track_name = "Track000{}".format(no)
    elif no in range(100, 1000):
        track_name = "Track00{}".format(no)
    elif no in range(1000, 2101):
        track_name = "Track0{}".format(no)
    return track_name

def detrackname(name):
    if name[6] != "0":
        no = name[6:]
    elif name[7] != "0":
        no = name[7:]
    elif name[8] != "0":
        no = name[8:]
    elif name[9] != "0":
        no = name[9:]
    return int(no)

def time2hms(time):
    hour = time//3600
    min = (time%3600)//60
    sec = (time%3600)%60
    if hour == 0:
        return f"{int(min):0>2}:{int(sec):0>2}"
    else:
        return f"{int(hour):0>2}:{int(min):0>2}:{int(sec):0>2}"

def bin2list(case, length):
    case = format(case, f"0{length}b") #2進数化
    return [int(i) for i in case]

def standardize(spec):
    std, mean = torch.std_mean(spec) #平均、標準偏差
    if -1e-4 < std and std < 1e-4:
        transformed = torch.zeros_like(spec) # 分散が0 = 元音源が無音 -> NaNを0に
        mean = torch.zeros_like(mean); std = torch.zeros_like(std) #分散、平均も0に
    else:
        transformed = (spec - mean) / std
    return mean, std, transformed

def destandardize(spec, mean, std):
    spec_ = torch.zeros_like(spec)
    for i in range(len(mean)):
        spec_[i] = spec[i] * std[i].item() + mean[i].item()
    return spec_

def normalize(spec, max = None, min = None):
    if max is None and min is None:
        max = torch.max(spec)
        min = torch.min(spec)
    if -1e-4 < max - min and max - min < 1e-4:
        transformed = torch.zeros_like(spec) # 分散が0 = 元音源が無音 -> NaNを0に
        max = torch.zeros_like(max); min = torch.zeros_like(min) #分散、平均も0に
    else:
        transformed = (spec - min) / (max - min)
    return max, min, transformed

def denormalize(max, min, spec):
    return spec * (max - min) + min

def l2normalize(vec):
    norm = np.linalg.norm(vec, ord=2)
    if norm <= 0.001:
        norm = 1.0
    return vec / norm


def nan_checker(array):
    """
    input ndarray
    '''
    return
    配列にNaNが含まれない   -> True
    配列にNaNが含まれる     -> False
    """
    if torch.any(torch.isnan(array)):
        return "Have NaN."
    else:
        return "Not Have NaN."

def start():
    if device == "cuda":
        torch.cuda.synchronize()
    start_time = time.time() # 時間計測開始
    return start_time

def finish(start_time):
    if device == "cuda":
        torch.cuda.synchronize()
    finish_time = time.time() - start_time # 時間計測終了
    return finish_time

def complex_norm(
        complex_tensor,
        power: float = 1.0
):
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    out = torch.view_as_real(complex_tensor).pow(2.).sum(-1).pow(power)
    return out

def angle(
        complex_tensor
):
    r"""Compute the angle of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`

    Return:
        Tensor: Angle of a complex tensor. Shape of `(..., )`
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])

def magphase_torch(
        complex_tensor,
        power: float = 1.0
):
    r"""Separate a complex-valued spectrogram with shape `(..., 2)` into its magnitude and phase.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`)

    Returns:
        (Tensor, Tensor): The magnitude and phase of the complex tensor
    """
    mag = torch.abs(complex_tensor)
    zeros_to_ones = torch.where(mag == 0, 1.0, 0.0)
    mag_nonzero = mag + zeros_to_ones
    phase = torch.empty_like(complex_tensor, dtype=torch.complex64, device=mag.device)
    phase.real = complex_tensor.real / mag_nonzero + zeros_to_ones # 無音ならphaseは1+j0
    phase.imag = complex_tensor.imag / mag_nonzero
    return mag, phase

def enhance_either_hpss(x_padded, out, kernel_size, power, which, offset):
    """x_padded: one that median filtering can be directly applied
    kernel_size = int
    dim: either 2 (freq-axis) or 3 (time-axis)
    which: str, either "harm" or "perc"
    """
    if which == "harm":
        for t in range(out.shape[3]):
            out[:, :, :, t] = torch.median(x_padded[:, :, offset:-offset, t:t + kernel_size], dim=3)[0]

    elif which == "perc":
        for f in range(out.shape[2]):
            out[:, :, f, :] = torch.median(x_padded[:, :, f:f + kernel_size, offset:-offset], dim=2)[0]
    else:
        raise NotImplementedError("it should be either but you passed which={}".format(which))

    if power != 1.0:
        out.pow_(power)


def hpss(x, kernel_size=31, power=2.0, hard=False):
    """x: |STFT| (or any 2-d representation) in batch, (not in a decibel scale!)
    in a shape of (batch, ch, freq, time)
    power: to which the enhanced spectrograms are used in computing soft masks.
    kernel_size: odd-numbered {int or tuple of int}
        if tuple,
            1st: width of percussive-enhancing filter (one along freq axis)
            2nd: width of harmonic-enhancing filter (one along time axis)
        if int,
            it's applied for both perc/harm filters
    """
    EPS = 1e-7
    eps = EPS
    if isinstance(kernel_size, tuple):
        pass
    else:
        # pad is int
        kernel_size = (kernel_size, kernel_size)

    pad = (kernel_size[0] // 2, kernel_size[0] // 2,
           kernel_size[1] // 2, kernel_size[1] // 2,)

    harm, perc, ret = torch.empty_like(x), torch.empty_like(x), torch.empty_like(x)
    x_padded = F.pad(x, pad=pad, mode='reflect')

    enhance_either_hpss(x_padded, out=perc, kernel_size=kernel_size[0], power=power, which='perc', offset=kernel_size[1]//2)
    enhance_either_hpss(x_padded, out=harm, kernel_size=kernel_size[1], power=power, which='harm', offset=kernel_size[0]//2)

    if hard:
        mask_harm = harm > perc
        mask_perc = harm < perc
    else:
        mask_harm = (harm + eps) / (harm + perc + eps)
        mask_perc = (perc + eps) / (harm + perc + eps)

    return x * mask_harm, x * mask_perc, mask_harm, mask_perc

def normalize_torch(data, max = None, min = None):
    if max is None:# and min is None:
        max = data.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    min = data.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    max_min = max - min
    max_min = torch.where(max_min == 0, 1, max_min) # 0なら1に変換
    transformed = (data - min) / max_min
    return transformed, max, min

def denormalize_torch(spec, max, min):
    return spec * (max - min) + min

def standardize_torch(data, dim=(1,2,3)):
    mean = data.mean(dim=dim, keepdim=True)
    std = data.std(dim=dim, keepdim=True)
    transformed = (data - mean) / (std + 1e-5)
    return transformed, mean, std

def destandardize_torch(data, mean, std):
    return data * std + mean

def get_fftfreq(
        sr: int = 44100,
        n_fft: int = 2048
) -> torch.Tensor:
    """
    Torch workaround of librosa.fft_frequencies
    srとn_fftから求められるstftの結果の周波数メモリを配列にして出力。
    0から始まり、最後が22050。
    """
    out = sr * torch.fft.fftfreq(n_fft)[:n_fft // 2 + 1]
    out[-1] = sr // 2
    return out

class TorchSTFT:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.melfilter = torch.from_numpy(np.squeeze(librosa.filters.mel(sr=cfg.sr, n_mels=cfg.n_mels, n_fft=cfg.f_size)))
        self.chromafilter = torch.from_numpy(librosa.filters.chroma(sr=cfg.sr, n_fft=cfg.f_size))
        self.fq_in_spec = get_fftfreq(sr=self.cfg.sr, n_fft=self.cfg.f_size)

    def _highlowpass_initialize(self, high_fq, low_fq):
        fq_in_spec = get_fftfreq(sr=self.cfg.sr, n_fft=self.cfg.f_size)
        """for n_fq, freq in enumerate(fq_in_spec):
            if freq >= low_fq:
                self.low_fq_idx = n_fq + 1
                break
        for n_fq, freq in enumerate(fq_in_spec):
            if freq >= high_fq:
                self.high_fq_idx = n_fq
                break"""
        self.low_fq_idx = low_fq
        self.high_fq_idx = high_fq
        
    def stft(self, wave):
        *other, length = wave.shape
        x = wave.reshape(-1, length)
        z = torch.stft(
                x,
                n_fft=self.cfg.f_size,
                hop_length=self.cfg.hop_length,
                window=torch.hann_window(self.cfg.f_size).to(wave),
                win_length=self.cfg.f_size,
                normalized=False,
                center=True,
                return_complex=True,
                pad_mode='reflect')
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)
    
    def highpass(self, spec):
        return spec[:, :, self.high_fq_idx:, :] # [B, C, F, T]を仮定
    
    def lowpass(self, spec):
        return spec[:, :, : self.low_fq_idx, :] # [B, C, F, T]を仮定

    def highlowpass(self, spec):
        return spec[:, :, self.high_fq_idx: self.low_fq_idx, :]
    
    def pitch_shift(self, wave, n_shift):
        return torchaudio.functional.pitch_shift(wave, self.cfg.sr, n_shift)
    
    def magphase(self, spec):
        transformed, phase = magphase_torch(spec)
        return transformed, phase
    
    def mel(self, spec):
        return torch.matmul(self.melfilter.to(spec.device), spec)

    def hpss_chroma(self, spec):
        spec_harm, spec_perc, _, _ = hpss(spec)
        return torch.einsum("cf,...ft->...ct", self.chromafilter.to(spec.device), spec_harm), spec_harm, spec_perc
    
    def amp2db(self, amp):
        return Fa.amplitude_to_DB(amp, 20, amin=1e-05, db_multiplier=0)

    def normalize(self, data, max = None, min = None):
        if max is None:# and min is None:
            max = data.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        min = data.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_min = max - min
        max_min = torch.where(max_min == 0, 1, max_min) # 0なら1に変換
        transformed = (data - min) / max_min
        return transformed, max, min

    def transform(self, sound, param=None):
        stft = self.stft(sound)
        if self.cfg.complex:
            *other, C, F, T = stft.shape
            transformed = torch.view_as_real(stft.reshape(-1, C, F, T)).permute(0, 1, 4, 2, 3).reshape(*other, C * 2, F, T)
            return transformed
        else:
            transformed, phase = magphase_torch(stft) #stftして振幅と位相に分解
            del sound
            if self.cfg.mel:
                transformed = torch.matmul(self.melfilter.to(transformed.device), transformed)
            if self.cfg.db:
                transformed = Fa.amplitude_to_DB(transformed, 20, amin=1e-05, db_multiplier=0)
        return transformed, phase
    
    def db2amp(self, spec):
        return Fa.DB_to_amplitude(spec, ref=1, power=0.5)
    
    def denormalize(self, spec, max, min):
        return spec * (max - min) + min

    def spec2complex(self, spec):
        B, C, F, T = spec.shape
        return torch.view_as_complex(spec.reshape(B, C // 2, 2, F, T).permute(0, 1, 3, 4, 2).contiguous()).reshape(B, C // 2, F, T)
    
    def istft(self, z):
        *other, freqs, frames = z.shape
        z = z.view(-1, freqs, frames)
        #print(z.dtype)
        x = torch.istft(z,
                    n_fft=self.cfg.f_size,
                    hop_length=self.cfg.hop_length,
                    window=torch.hann_window(self.cfg.f_size).to(z.real),
                    win_length=self.cfg.f_size,
                    normalized=False,
                    #length=length,
                    center=True)
        _, length = x.shape
        return x.view(*other, length)

    def detransform(self, spec, phase = None):
        #print(spec.shape)
        #spec_denormal = self.denormalize(spec, max, min).to("cpu").numpy() #正規化を解除
        if self.cfg.complex:
            #*other, C, F, T = spec.shape
            B, C, F, T = spec.shape
            z = torch.view_as_complex(spec.reshape(B, C // 2, 2, F, T).permute(0, 1, 3, 4, 2).contiguous()).reshape(B, C // 2, F, T)
        else:
            if self.cfg.db:
                spec = Fa.DB_to_amplitude(spec, ref=1, power=0.5) #dbを元の振幅に直す
            z = spec * phase
        *other, freqs, frames = z.shape
        #n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        #print(z.dtype)
        x = torch.istft(z,
                    n_fft=self.cfg.f_size,
                    hop_length=self.cfg.hop_length,
                    window=torch.hann_window(self.cfg.f_size).to(z.real),
                    win_length=self.cfg.f_size,
                    normalized=False,
                    #length=length,
                    center=True)
        _, length = x.shape
        return x.view(*other, length)

def file_exist(dir_path):
    # ディレクトリがない場合、作成する
    if not os.path.exists(dir_path):
        print("ディレクトリを作成します")
        os.makedirs(dir_path)

def evaluate(reference, estimate, inst_list, writer, epoch):
    # assume mix as estimates
    B, C, S, T = reference.shape
    reference = torch.reshape(reference, (B, T, C*S))
    estimate  = torch.reshape(estimate, (B, T, C*S))
    scores = {}
    for inst in inst_list:
        scores[inst] = {"SDR":0, "ISR":0, "SIR":0, "SAR":0}
    for idx, inst in enumerate(inst_list):
        # Evaluate using museval
        score = museval.evaluate(references=reference[:,:,idx*S:(idx+1)*S], estimates=estimate[:,:,idx*S:(idx+1)*S])
        #print(score)
        for i,key in enumerate(list(scores[inst].keys())):
            scores[inst][key] = np.mean(score[i])
    return scores

def knn_psd(label:np.ndarray, vec:np.ndarray, cfg, psd: bool):
    knn_start = start()
    print(f"= kNN...")
    total_all   = 0
    correct_all = 0
    knn_sk = KNeighborsClassifier(n_neighbors=5, weights="uniform", n_jobs=cfg.num_workers)
    for idx in range(len(label)):
        if psd: # 擬似楽曲の時
            reduce_idx = np.where((label[:,0]==label[idx,0]) & (label[:,1]==label[idx,1]))[0].tolist() # fitする擬似楽曲と構成が同じ曲を除く
            knn_sk.fit(np.delete(vec, reduce_idx, axis=0), np.delete(label[:,0], reduce_idx, axis=0))
        else: # 擬似楽曲でない曲の時
            knn_sk.fit(np.delete(vec, idx, axis=0), np.delete(label[:,0], idx, axis=0)) # 推測するsegのみ削除する。
        pred = knn_sk.predict(vec[idx].reshape(1, -1))
        if label[idx,0] == pred:
            correct_all += 1
        total_all += 1
    knn_time = finish(knn_start)
    print(f"kNN was finished!")
    print(f"= kNN time is {knn_time} sec. =")
    return correct_all / total_all

def tsne_psd(label:np.ndarray, vec:np.ndarray, mode: str, cfg, dir_path:str, current_epoch=0):
    # TODO:このTSNEを使うときはdataloaderのshuffle=Falseにする。
    tsne_start = start()
    print(f"= T-SNE...")
    counter = 0
    num_continue = 10
    markers = [",", "o", "v", "^", "p", "D", "<", ">", "8", "*"]
    colors = ["r", "g", "b", "c", "m", "y", "k", "#ffa500", "#00ff00", "gray"]
    num_songs = 10
    color10 = []
    marker10 = []
    vec10 = []
    id_all = np.unique(label[:, 0])
    #while counter < num_songs:
    for n, picked_id in enumerate(id_all):
        samesong_idx = np.where(label[:,0]==picked_id)[0]
        samesong_vec = vec[samesong_idx]
        samesong_label = label[samesong_idx]
        # 色を指定
        color10 = color10 + [colors[n] for i in range(samesong_idx.shape[0])]
        # マークを指定
        counter_m = -1 # 便宜上。本当は0にしたい
        log_ver = []
        for i in range(samesong_idx.shape[0]):
            if not samesong_label[i, 1] in log_ver:
                log_ver.append(samesong_label[i, 1])
                counter_m += 1
            marker10.append(markers[counter_m])
        vec10.append(samesong_vec)
    vec10 = np.concatenate(vec10, axis=0)
    perplexity = [5, 15, 30, 50]
    for i in range(len(perplexity)):
        fig, ax = plt.subplots(1, 1)
        X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i], n_jobs=cfg.num_workers).fit_transform(vec10)
        for j in range(len(vec10)):
            mappable = ax.scatter(X_reduced[j, 0], X_reduced[j, 1], c=color10[j], marker=marker10[j], s=30)
        file_exist(dir_path)
        fig.savefig(dir_path + f"/psd_emb_{mode}_e{current_epoch}_s{counter}_tsne_p{perplexity[i]}_m{cfg.margin}.png")
        plt.clf()
        plt.close()
    tsne_time = finish(tsne_start)
    print(f"T-SNE was finished!")
    print(f"= T-SNE time is {tsne_time} sec. =")

def tsne_not_psd(label:np.ndarray, vec:np.ndarray, mode: str, cfg, dir_path:str, current_epoch=0):
    tsne_start = start()
    print(f"= T-SNE...")
    num_songs = 20 if cfg.n_song_test >= 20 else cfg.n_song_test
    color10 = []
    vec10 = []
    id_all = np.unique(label[:, 0])
    id_picked = np.random.choice(id_all, num_songs, replace=False)
    for n, picked_id in enumerate(id_picked):
        samesong_idx = np.where(label[:,0]==picked_id)[0]
        samesong_vec = vec[samesong_idx]
        # 色を指定
        color10 = color10 + [n for _ in range(samesong_idx.shape[0])] # 色番号のみ格納
        # マークを指定
        vec10.append(samesong_vec)
    vec10 = np.concatenate(vec10, axis=0)
    perplexity = [5, 15, 30, 50]
    for i in range(len(perplexity)):
        fig, ax = plt.subplots(1, 1)
        X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i], n_jobs=cfg.num_workers).fit_transform(vec10)
        mappable = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=color10, s=30, cmap="tab20")
        ax.legend(mappable.legend_elements(num=num_songs)[0], id_picked, borderaxespad=0, bbox_to_anchor=(1.05, 1),
                    loc="upper left", title="Songs")
        file_exist(dir_path)
        fig.savefig(dir_path + f"/not_psd_emb_{mode}_e{current_epoch}_s{num_songs}_tsne_p{perplexity[i]}_m{cfg.margin}.png", bbox_inches='tight')
        plt.clf()
        plt.close()
    tsne_time = finish(tsne_start)
    print(f"T-SNE was finished!")
    print(f"= T-SNE time is {tsne_time} sec. =")


def tsne_psd_marker(label:np.ndarray, vec:np.ndarray, mode: str, cfg, dir_path:str, current_epoch=0):
    tsne_start = start()
    print(f"= T-SNE...")
    counter = 0
    markers = [",", "o", "v", "^", "p", "D", "<", ">", "8", "*"]
    colors = ["r", "g", "b", "c", "m", "y", "k", "#ffa500", "#00ff00", "gray"]
    id_list = []
    ver_list = []
    for id in label[:,0]:
        if not id in id_list:
            id_list.append(id)
    for ver in label[:,1]:
        if not ver in ver_list:
            ver_list.append(ver)
    perplexity = [5, 15, 30, 50]
    for i in range(len(perplexity)):
        fig, ax = plt.subplots(1, 1)
        X_reduced = TSNE(n_components=2, random_state=0, perplexity=perplexity[i], n_jobs=cfg.num_workers).fit_transform(vec)
        for j in range(len(vec)):
            mappable = ax.scatter(X_reduced[j, 0], X_reduced[j, 1], c=colors[id_list.index(label[j,0])], marker=markers[ver_list.index(label[j,1])], s=30)
        file_exist(dir_path)
        fig.savefig(dir_path + f"/emb_{mode}_e{current_epoch}_s{counter}_tsne_p{perplexity[i]}_m{cfg.margin}.png")
        plt.clf()
        plt.close()
    tsne_time = finish(tsne_start)
    print(f"T-SNE was finished!")
    print(f"= T-SNE time is {tsne_time} sec. =")


if "__main__" == __name__:
    pass
from typing import Any
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import librosa
import random

from utils.func import stft, trackname, detrackname, l2normalize, STFT
from model.csn import ConditionalSimNet1d31cases, ConditionalSimNet1d
from .dataset_triplet import BA4Track, BA4Track_31ways, B4TrackInst

def return_second_offset(cfg, mode, datasettype):
    if datasettype == "psd":
        if mode == "train":
            second = cfg.seconds_psd_train
            offset = cfg.offset_psd_train
        elif mode == "valid":
            second = cfg.seconds_psd_valid
            offset = cfg.offset_psd_valid
        elif mode == "test":
            second = cfg.seconds_psd_test
            offset = cfg.offset_psd_test
    elif datasettype == "not_psd":
        if mode == "train":
            second = cfg.seconds_not_psd_train
            offset = cfg.offset_not_psd_train
        elif mode == "valid":
            second = cfg.seconds_not_psd_valid
            offset = cfg.offset_not_psd_valid
        elif mode == "test":
            second = cfg.seconds_not_psd_test
            offset = cfg.offset_not_psd_test
    elif datasettype == "triplet":
        if mode == "train":
            second = cfg.seconds_triplet_train
            offset = cfg.offset_triplet_train
        elif mode == "valid":
            second = cfg.seconds_triplet_valid
            offset = cfg.offset_triplet_valid
        elif mode == "test":
            second = cfg.seconds_triplet_test
            offset = cfg.offset_triplet_test
    elif datasettype == "c32":
        if mode == "train":
            second = cfg.seconds_c32_train
            offset = cfg.offset_c32_train
        elif mode == "valid":
            second = cfg.seconds_c32_valid
            offset = cfg.offset_c32_valid
        elif mode == "test":
            second = cfg.seconds_c32_test
            offset = cfg.offset_c32_test
    elif datasettype == "triplet_eval":
        if mode == "train":
            second = cfg.seconds_triplet_eval_train
            offset = cfg.offset_triplet_eval_train
        elif mode == "valid":
            second = cfg.seconds_triplet_eval_valid
            offset = cfg.offset_triplet_eval_valid
        elif mode == "test":
            second = cfg.seconds_triplet_eval_test
            offset = cfg.offset_triplet_eval_test
    return second, offset

class ReturnBPM:
    def __init__(self, cfg) -> None:
        self.bpm_list = pd.read_csv(cfg.metadata_dir + "slakh/tempo2100mix.csv").values
    
    def __call__(self, track_id) -> Any:
        return self.bpm_list[track_id - 1, 1]

def loadseg_from_npz(path):
    npz = np.load(path)
    return npz["wave"].astype(np.float32), npz["sound"]

class LoadSegZume:
    def __init__(self, cfg, mode:str, second, offset, dirpath) -> None:
        self.mode = mode
        self.cfg = cfg
        self.dirpath = dirpath
        self.second = second
        self.offset = offset
    
    def load(self, track_id, seg_id, inst, n_shift):
        if self.cfg.load_using_librosa:
            stem_path = f"/nas03/assets/Dataset/slakh-2100_2/{trackname(track_id)}/submixes/{inst}.wav"
            stem_wave, sr = librosa.load(
                stem_path,
                sr=self.cfg.sr,
                mono=self.cfg.mono,
                offset=seg_id*self.offset,
                duration=self.second)
        else:
            path = self.dirpath + f"/{inst}/wave{track_id}_{seg_id}.npz"
            stem_wave, sound = loadseg_from_npz(path=path)
        if self.cfg.pitch_shift:
            #n_shift = random.sample([-1, 0, 1], 1)[0]
            if n_shift != 0:
                stem_wave = librosa.effects.pitch_shift(stem_wave, sr=self.cfg.sr, n_steps=n_shift)
        return stem_wave

def max_std(x, axis=None):
    max = abs(x).max(axis=axis, keepdims=True) + 0.000001
    result = x / max
    return result

class CreatePseudo(Dataset):
    """tracklistの曲のseglistのセグメントを読み込んで、擬似楽曲を完成させる。"""
    def __init__(
        self,
        cfg,
        datasettype,
        mode="train",
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.second, self.offset = return_second_offset(cfg, mode, datasettype)
        dirpath = f"{self.cfg.dataset_dir}/cutwave/{self.second}s_on{self.offset}"
        self.loader = LoadSegZume(self.cfg, mode, self.second, self.offset, dirpath)
    
    def _basic_aditional_to_inst(self, c):
        c_inst = []
        if type(c) == list:
            for i in c:
                c_inst.append(self.cfg.inst_all[i.item()])
        elif type(c) == int:
            c_inst.append(self.cfg.inst_all[c])
        else:
            raise NotImplementedError
        return c_inst

    def load_mix_stems(self, tracklist, seglist, c=[], condition=None, track="none"):
        if not c:
            c_inst = []
        else:
            c_inst = self._basic_aditional_to_inst(c)
        if condition is None:
            condition = [1 for i in range(len(self.cfg.inst_all))]
        #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
        stems = []
        if self.cfg.load_using_librosa:
            mix = np.zeros((1, self.second*self.cfg.sr), dtype=np.float32)
        else:
            if self.second == 3:
                mix = np.zeros((1, 131072), dtype=np.float32)
            elif self.second == 10:
                mix = np.zeros((1, 440320), dtype=np.float32)
        #mix = np.zeros((1, 131072), dtype=np.float32)
        if track == "positive":
            n_shift = random.sample([i for i in range(-self.cfg.n_shift, self.cfg.n_shift + 1)], 1)[0]
        else:
            n_shift = 0
        for iter, inst in enumerate(self.cfg.inst_all):
            if condition[iter] == 1:
                stem_wave = self.loader.load(tracklist[iter], seglist[iter], inst, n_shift=n_shift if inst in c_inst else 0)
                if len(stem_wave) < mix.shape[1] and self.cfg.load_using_librosa:
                    stem_wave = np.pad(stem_wave, (0, mix.shape[1] - len(stem_wave)))
                if self.cfg.mono:
                    stem_wave = np.expand_dims(stem_wave, axis=0)
                mix = mix[:,:stem_wave.shape[1]] #なぜかshapeが違う。mixは132300(3×44100)なのにstem_waveは131072。なぜ？
                mix += stem_wave; stems.append(stem_wave)
            else:
                if self.cfg.load_using_librosa:
                    stems.append(np.zeros((1, self.second*self.cfg.sr), dtype=np.float32))
                else:
                    if self.second == 3:
                        stems.append(np.zeros((1, 131072), dtype=np.float32))
                    elif self.second == 10:
                        stems.append(np.zeros((1, 440320), dtype=np.float32))
        stems = np.stack(stems, axis=0)
        return mix, stems

class SongDataForPreTrain(Dataset):
    """事前学習用のdataset"""
    def __init__(
        self,
        cfg,
        mode="train"
        ) -> None:
        super().__init__()
        self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/single3_200data-euc_zero.csv", index_col=0).values
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)
        self.cfg = cfg
    
    def load_embvec(self, track_id, seg_id, condition=None):
        #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
        if condition is None:
            condition = [1 for _ in range(len(self.cfg.inst_all))]
        dirpath = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/"
        embvec = []
        for idx, inst in enumerate(self.cfg.inst_all):
            if condition[idx] == 1:
                vec_inst = np.load(dirpath + inst + f"/Track{track_id}/seg{seg_id}.npy")
                #print(inst, np.max(vec_inst))
            else:
                vec_inst = np.zeros((1,128), dtype=np.float32)
            if self.cfg.normalize128:
                vec_inst = l2normalize(vec_inst)
            embvec.append(vec_inst)
        return torch.from_numpy(l2normalize(np.concatenate(embvec, axis=1).squeeze()))

    def load_embvec_640(self, track_id, seg_id, condition=0b11111):
        """全ての音源のembを読んで640次元で正規化してからcsnでconditionの部分だけ残してreturn。"""
        #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
        dirpath = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/"
        embvec = self.load_embvec(track_id, seg_id)
        csn = ConditionalSimNet1d31cases(self.cfg)
        return csn(embvec, torch.tensor([condition], device=embvec.device))
    
    def load_embvec_stems(self, track_id, seg_id, condition=None):
        if self.cfg.emb_640norm:
            embvec = self.load_embvec(track_id, seg_id)
            csn = ConditionalSimNet1d()
            embvec_inst = []
            for idx,inst in enumerate(self.cfg.inst_all):
                embvec_inst.append(csn(embvec, torch.tensor([idx], device=embvec.device)))
            return torch.stack(embvec_inst, dim=0).squeeze()
            #return torch.stack([csn(embvec, torch.tensor([i], device=embvec.device)) for i in range(len(self.cfg.inst_list))], dim=0).squeeze()
        else:
            if condition is None:
                condition = [1 for i in range(len(self.cfg.inst_all))]
            #bin_str = format(condition, f"0{len(self.cfg.inst_list)}b") #2進数化
            dirpath = f"/nas03/assets/Dataset/slakh/single3_200data-euc_zero/"
            embvec = []
            for idx, inst in enumerate(self.cfg.inst_all):
                vec_inst = np.zeros((1,640), dtype=np.float32)
                if condition[idx] == 1:
                    vec_inst[:,idx*128:(idx+1)*128] = np.load(dirpath + inst + f"/Track{track_id}/seg{seg_id}.npy")
                else:
                    pass
                    #vec_inst = np.zeros((1,128), dtype=np.float32)
                #if self.cfg.normalize128:
                #    vec_inst = l2normalize(vec_inst)
                embvec.append(l2normalize(vec_inst.squeeze()))
            return torch.from_numpy(np.stack(embvec, axis=0))
    
    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, index):
        track_id, seg_id = self.datafile[index, 0], self.datafile[index, 1]
        tracklist = [track_id for _ in self.cfg.inst_all]
        seglist   = [seg_id   for _ in self.cfg.inst_all]
        if self.cfg.condition32:
            condition_b = random.randrange(0, 2**len(self.cfg.inst_all))
            condition = [int(i) for i in format(condition_b, f"0{len(self.cfg.inst_all)}b")]
            mix_spec, _ = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
            if self.cfg.emb_640norm:
                return mix_spec, condition, self.load_embvec_640(track_id, seg_id, condition=condition_b).squeeze()
            else:
                return mix_spec, condition, self.load_embvec(track_id, seg_id, condition=condition)
        else:
            mix_spec, stems_spec = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist)
            return mix_spec, stems_spec, self.load_embvec(track_id, seg_id), self.load_embvec_stems(track_id, seg_id)

def load_lst(listpath):
    data = []
    #listdir = cfg.metadata_dir + "lst"
    #with open(f"{listdir}/{listname}", "r") as f:
    with open(listpath, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            groups = line.split(";")
            items = []
            for group in groups:
                numbers = [int(num.strip()) for num in group.split(",")]
                items.append(numbers)
            data.append(tuple(items))
    return data

class PsdLoader(Dataset):
    """test用擬似楽曲をロード"""
    def __init__(
        self,
        cfg,
        inst,
        mode:str,
        psd_mine: bool
    ):
        lst_dir = cfg.metadata_dir + f"lst/"
        self.cfg = cfg
        #if cfg.test_psd_mine:
        second, offset = return_second_offset(cfg, mode, "psd")
        if psd_mine:
            self.data = load_lst(lst_dir + f"psd_take_{mode}_{second}s_ol{float(second)}/{inst}.lst")
        else:
            #TODO:今自作psdをやめてます．
            #self.data = load_lst(lst_dir + f"psds_{mode}_{cfg.n_song_psd}songs_{second}s_ol{float(second)}/{inst}.lst")
            if mode == "valid":
                self.data = load_lst(lst_dir + f"psds_{mode}_10_{inst}.lst")
            elif mode == "test":
                self.data = load_lst(lst_dir + f"psds_{mode}_{inst}.lst")
        self.loader = CreatePseudo(cfg, datasettype="psd", mode=mode)
        self.condition = cfg.inst_all.index(inst)
        self.mode = mode
        self.bpmer = ReturnBPM(cfg)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        c_selected, tracklist, seglist, ID_ver = self.data[index]
        ID = ID_ver[0]; ver = ID_ver[1]
        #ver = f"{ID}-{ver}"
        #if self.mode == "test":
        #    data, _, _, _ = self.loader.load_mix_stems(tracklist, seglist)
        #else:
        #    data, _ = self.loader.load_mix_stems(tracklist, seglist)
        data_psd, _ = self.loader.load_mix_stems(tracklist, seglist)
        """data_not_psd, _ = self.loader.load_mix_stems(
            tracklist=[tracklist[c_selected[0]] for i in range(len(self.cfg.inst_all))],
            seglist=[seglist[c_selected[0]] for i in range(len(self.cfg.inst_all))])"""
        bpm = [self.bpmer(id) for id in tracklist]
        return ID, ver, seglist[c_selected[0]], data_psd, torch.FloatTensor(bpm), self.condition

class TestLoader(Dataset):
    """test用擬似楽曲でない曲をロード"""
    def __init__(
        self,
        cfg,
        inst,
        mode:str,
    ):
        lst_dir = cfg.metadata_dir + f"zume/slakh/"
        self.cfg = cfg
        self.condition = cfg.inst_all.index(inst)
        self.mode = mode
        self.data = self.make_test_data(inst, mode)
        self.loader = CreatePseudo(cfg, datasettype="not_psd", mode=mode)
        self.bpmer = ReturnBPM(cfg)
    
    def make_test_data(self, inst, mode):
        lst_dir = self.cfg.metadata_dir + f"zume/slakh/"
        second, offset = return_second_offset(self.cfg, mode, "not_psd")
        if self.cfg.not_psd_no_silence_stem:
            metadata = pd.read_csv(self.cfg.metadata_dir + f"slakh/{second}s_no_silence_or{offset}_0.25/{mode}_slakh_{second}s_stem_normal.csv").values
        else:
            metadata = pd.read_csv(self.cfg.metadata_dir + f"slakh/{second}s_no_silence_or{offset}_0.25/{mode}_slakh_{second}s_silence_data_normal.csv").values
        if mode == "test":
            songdata = json.load(open(self.cfg.metadata_dir + f"slakh/test_redux_136.json", 'r'))
            picked_id = songdata
        elif mode == "valid":
            songdata = json.load(open(self.cfg.metadata_dir + f"slakh/valid_redux.json", 'r'))
            if self.cfg.not_psd_all:
                picked_id = songdata
            else:
                picked_id = random.sample(songdata, self.cfg.n_song_test)
        data = []
        for id in picked_id:
            #print(id, type(id))
            # +3は、silence_dataファイルの最初3列のtrack_id,seg_id,mix_silenceを飛ばすため
            if self.cfg.not_psd_no_silence_stem:
                track_all = metadata[np.where(metadata[:, 0] == id)[0]]
            else:
                track_all = metadata[np.where((metadata[:, 0] == id) & (metadata[:, self.cfg.inst_all.index(inst) + 3] == 1))[0]]
            for track in track_all:
                data.append(
                    [
                        [self.cfg.inst_all.index(inst)], # c_selected
                        [track[0] for _ in self.cfg.inst_all], # tracklist
                        [track[1] for _ in self.cfg.inst_all], # seglist
                        [track[0], track[0]] # ID_ver
                    ]
                )
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        c_selected, tracklist, seglist, ID_ver = self.data[index]
        ID = ID_ver[0]; ver = ID_ver[1]
        #ver = f"{ID}-{ver}"
        #if self.mode == "test":
        #    data, _, _, _ = self.loader.load_mix_stems(tracklist, seglist)
        #else:
        #    data, _ = self.loader.load_mix_stems(tracklist, seglist)
        data_psd, _ = self.loader.load_mix_stems(tracklist, seglist)
        bpm = [self.bpmer(id) for id in tracklist]
        #return ID, ver, seglist[c_selected[0]], data_psd, torch.FloatTensor(bpm), self.condition
        return ID, seglist[c_selected[0]], seglist[c_selected[0]], data_psd, torch.FloatTensor(bpm), self.condition

class TripletLoaderFromList(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        inst=None,
        ):
        self.cfg = cfg
        lst_dir = cfg.metadata_dir + f"lst/"
        if mode == "train":
            #self.triplets = load_lst(f"triplets_1200_ba1_4_withsil_20000triplets.lst")
            if self.cfg.pseudo == "ba_4t":
                self.triplets = load_lst(lst_dir + f"triplets_1200_ba4t_withsil_nosegsfl_20000triplets.lst")
            elif self.cfg.pseudo == "31ways":
                self.triplets = load_lst(lst_dir + f"triplets_ba4t_31ways_train_3s_10000songs.lst")
            elif self.cfg.pseudo == "b_4t_inst":
                self.triplets = load_lst(lst_dir + f"triplets_b4t_{inst}_train_3s_10000songs.lst")
        elif mode == "valid":
            #self.triplets = load_lst(f"triplets_valid_ba1_4_withsil_200triplets.lst")
            if self.cfg.pseudo == "ba_4t":
                self.triplets = load_lst(lst_dir + f"triplets_valid_ba4t_withsil_nosegsfl_200triplets.lst")
            elif self.cfg.pseudo == "31ways":
                self.triplets = load_lst(lst_dir + f"triplets_ba4t_31ways_valid_10s_2000songs.lst")
            elif self.cfg.pseudo == "b_4t_inst":
                self.triplets = load_lst(lst_dir + f"triplets_b4t_{inst}_valid_10s_200songs.lst")
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        # print(self.triplets[index])
        (
            tracklist_a,
            tracklist_p,
            tracklist_n,
            seglist_a,
            seglist_p,
            seglist_n,
            sound_a,
            sound_p,
            sound_n,
            c,
        ) = self.triplets[index]
        mix_a, stems_a = self.loader.load_mix_stems(tracklist_a, seglist_a, sound_a)
        mix_p, stems_p = self.loader.load_mix_stems(tracklist_p, seglist_p, sound_p)
        mix_n, stems_n = self.loader.load_mix_stems(tracklist_n, seglist_n, sound_n)
        # cのindexにおいてanchor、positive、negativeの曲のセグメント
        return mix_a, stems_a, mix_p, stems_p, mix_n, stems_n, torch.FloatTensor(sound_a), torch.FloatTensor(sound_p), torch.FloatTensor(sound_n), c[0]

class TripletLoaderEachTime(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        inst=None,
        ):
        self.cfg = cfg
        if mode == "train":
            self.n_triplet = self.cfg.n_triplet_train
        elif mode == "valid":
            self.n_triplet = self.cfg.n_triplet_valid
        if self.cfg.pseudo == "ba_4t":
            self.triplet_maker = BA4Track(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        elif self.cfg.pseudo == "31ways":
            self.triplet_maker = BA4Track_31ways(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        elif self.cfg.pseudo == "b_4t_inst":
            self.triplet_maker = B4TrackInst(n_triplet=self.n_triplet, cfg=cfg, mode=mode)
        else:
            raise NotImplementedError
        self.loader = CreatePseudo(cfg, datasettype="triplet", mode=mode)
        self.bpmer = ReturnBPM(cfg)

    def __len__(self):
        return self.n_triplet

    def __getitem__(self, index):
        # print(self.triplets[index])
        (
            tracklist_a,
            tracklist_p,
            tracklist_n,
            seglist_a,
            seglist_p,
            seglist_n,
            sound_a,
            sound_p,
            sound_n,
            c, # [b, a]の順で入っている
        ) = self.triplet_maker()
        mix_a, stems_a = self.loader.load_mix_stems(tracklist_a, seglist_a, condition=sound_a, c=c, track="anchor")
        mix_p, stems_p = self.loader.load_mix_stems(tracklist_p, seglist_p, condition=sound_p, c=c, track="positive")
        mix_n, stems_n = self.loader.load_mix_stems(tracklist_n, seglist_n, condition=sound_n, c=c, track="negative")
        bpm_a = [self.bpmer(id) for id in tracklist_a]
        bpm_p = [self.bpmer(id) for id in tracklist_p]
        bpm_n = [self.bpmer(id) for id in tracklist_n]
        # cのindexにおいてanchor、positive、negativeの曲のセグメント
        return (mix_a, stems_a, mix_p, stems_p, mix_n, stems_n,
                torch.FloatTensor(sound_a), torch.FloatTensor(sound_p), torch.FloatTensor(sound_n),
                torch.FloatTensor(bpm_a), torch.FloatTensor(bpm_p), torch.FloatTensor(bpm_n),
                torch.tensor(c))

class TripletLoaderAllDiff(Dataset):
    def __init__(self,
        cfg,
        mode,
        inst=None,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.second, self.offset = return_second_offset(cfg, mode, datasettype="triplet")
        self.bpmer = ReturnBPM(cfg)
        self.inst = inst
        if cfg.load_using_librosa:
            self.datafile = pd.read_csv(cfg.metadata_dir + f"slakh/{self.second}s_no_silence_or{self.offset}_0.25/{mode}_slakh_{self.second}s_stem_normal.csv").values
        else:
            assert False, "cfg.load_using_librosa is False."
            self.datafile = pd.read_csv(cfg.metadata_dir + f"zume/slakh/{self.second}s_on{self.offset}_in_sametempolist_{mode}.csv").values
    
    def __len__(self):
        if self.mode == "train":
            return self.cfg.n_triplet_train
        elif self.mode == "valid":
            return self.cfg.n_triplet_valid
    
    def _load_sound(self, track, seg, mix: bool):
        # mix音源
        if mix:
            path = "/nas03/assets/Dataset/slakh-2100_2" + "/" + trackname(track) + "/mix.flac"
        else:
            path = f"/nas03/assets/Dataset/slakh-2100_2/{trackname(track)}/submixes/{self.inst}.wav"
        wave, _ = librosa.load(
            path,
            sr=self.cfg.sr,
            mono=self.cfg.mono,
            offset=seg * self.offset,
            duration=self.second)
        if self.cfg.mono:
            wave = np.expand_dims(wave, axis=0)
        return wave
    
    def __getitem__(self, index) -> Any:
        # 重複ありでapnのインデックス出力
        r = [random.randrange(len(self.datafile)) for _ in range(3)]
        mix_a = self._load_sound(self.datafile[r[0], 0], self.datafile[r[0], 1], mix=True)
        mix_p = self._load_sound(self.datafile[r[1], 0], self.datafile[r[1], 1], mix=True)
        mix_n = self._load_sound(self.datafile[r[2], 0], self.datafile[r[2], 1], mix=True)
        stem_a = self._load_sound(self.datafile[r[0], 0], self.datafile[r[0], 1], mix=False)
        stem_p = self._load_sound(self.datafile[r[1], 0], self.datafile[r[1], 1], mix=False)
        stem_n = self._load_sound(self.datafile[r[2], 0], self.datafile[r[2], 1], mix=False)
        bpm_a = self.bpmer(self.datafile[r[0], 0])
        bpm_p = self.bpmer(self.datafile[r[1], 0])
        bpm_n = self.bpmer(self.datafile[r[2], 0])
        bpm_a = torch.FloatTensor([bpm_a])
        sound = torch.ones(5) # 全部有音として扱う
        return (mix_a, stem_a, mix_p, stem_p, mix_n, stem_n,
                sound, sound, sound,
                bpm_a, torch.FloatTensor([bpm_p]), torch.FloatTensor([bpm_n]),
                torch.full_like(bpm_a, fill_value=self.cfg.inst_all.index(self.cfg.inst), dtype=torch.int).squeeze()
                )

class Condition32Loader(Dataset):
    def __init__(
        self,
        cfg,
        mode,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        if mode == "train":
            self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/3s_on1.5_no_silence_all.csv", index_col=0).values[:104663]
        elif mode == "valid":
            self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/10s_on5.0_no_silence_all.csv", index_col=0).values[44683:]
        elif mode == "test":
            self.datafile = pd.read_csv(cfg.metadata_dir + "zume/slakh/10s_on10.0_no_silence_all.csv", index_col=0).values
        self.loader = CreatePseudo(cfg, datasettype="c32", mode=mode)
        self.mode = mode
    
    def __len__(self):
        if self.mode == "train" or self.mode == "valid":
            return len(self.datafile)
        elif self.mode == "test":
            return self.cfg.n_dataset_test
    
    def __getitem__(self, index):
        track_id, seg_id = self.datafile[index, 0], self.datafile[index, 1]
        tracklist = [track_id for _ in self.cfg.inst_all]
        seglist   = [seg_id   for _ in self.cfg.inst_all]
        condition = random.randrange(0, 2**len(self.cfg.inst_all))
        cases = format(condition, f"0{len(self.cfg.inst_all)}b") #2進数化
        condition = [int(i) for i in cases]
        cases = torch.tensor([float(int(i)) for i in cases])
        #if self.mode == "train" or self.mode == "valid":
        #    mix_spec, stems_spec = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
        #    return mix_spec, stems_spec, cases
        #elif self.mode == "test":
        #    mix_spec, stems_spec, param, phase = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
        #    return mix_spec, stems_spec, param, phase, cases
        mix, stems = self.loader.load_mix_stems(tracklist=tracklist, seglist=seglist, condition=condition)
        return mix, stems, cases

class NormalSongLoaderForMOSZume2023(Dataset):
    def __init__(self, cfg, inst) -> None:
        super().__init__()
        self.cfg = cfg
        self.path = "/nas03/assets/Dataset/slakh-2100_2"
        dirpath = "/nas01/homes/imamura23-1000067/codes/MusicSimilarityWithUNet/mos_zume_2023/soundfile"
        set1 = pd.read_csv(dirpath + f"/{inst}_set1.csv").values
        set2 = pd.read_csv(dirpath + f"/{inst}_set2.csv").values
        self.datafile = np.concatenate([set1, set2], axis=0)
        self.second = cfg.seconds_not_psd_test
        self.offset = cfg.offset_not_psd_test
        self.condition = cfg.inst_all.index(inst)
        self.bpmer = ReturnBPM(cfg)
    
    def __len__(self):
        return len(self.datafile)
    
    def __getitem__(self, idx) -> Any:
        n = int(self.datafile[idx][1])
        # mix音源
        mix_path = self.path + "/" + trackname(int(self.datafile[idx][0])) + "/mix.flac"
        mix_wave, _ = librosa.load(
            mix_path,
            sr=self.cfg.sr,
            mono=self.cfg.mono,
            offset=n*self.offset,
            duration=self.second)
        if self.cfg.mono:
            mix_wave = np.expand_dims(mix_wave, axis=0)
        bpm = [self.bpmer(int(self.datafile[idx][0])) for inst in self.cfg.inst_all]
        return self.datafile[idx][0], self.datafile[idx][1], self.datafile[idx][1], mix_wave, torch.FloatTensor(bpm), self.condition

class DataLoaderForABXZume2024(Dataset):
    def __init__(self, cfg, mode, inst) -> None:
        super().__init__()
        self.cfg = cfg
        self.inst = inst
        self.condition = cfg.inst_all.index(inst)
        self.bpmer = ReturnBPM(cfg)
        #self.path = "/nas03/assets/Dataset/slakh-2100_2"
        self.n_bits = 219136
        self.results_all = json.load(open(self.cfg.metadata_dir + f"zume/abx_2024/results_modified.json", 'r'))
        self.identifier = []
        self.results = {}
        for key, value in self.results_all.items():
            if value["inst"] == inst:
                self.results[key] = value
                self.identifier.append(key)


    def __len__(self):
        return len(self.identifier)
    
    def _load(self, track, seg):
        mix_wave = np.zeros((1, 219136), dtype=np.float32)
        for inst in self.cfg.inst_all:
            path = f"{self.cfg.dataset_dir}/cutwave/5s_on5.0/{inst}/wave{track}_{seg}.npz"
            stem_wave, sound = loadseg_from_npz(path=path)
            if self.cfg.mono:
                stem_wave = np.expand_dims(stem_wave, axis=0)
            mix_wave += stem_wave
        return mix_wave
    
    def __getitem__(self, index) -> Any:
        idnt = self.identifier[index]
        result = self.results[idnt]
        result_id = result["id"]
        result_seg = result["seg"]
        mix_x = self._load(result_id["X"], result_seg["X"])
        mix_a = self._load(result_id["A"], result_seg["A"])
        mix_b = self._load(result_id["B"], result_seg["B"])
        bpm_x = [self.bpmer(result_id["X"]) for inst in self.cfg.inst_all]
        bpm_a = [self.bpmer(result_id["A"]) for inst in self.cfg.inst_all]
        bpm_b = [self.bpmer(result_id["B"]) for inst in self.cfg.inst_all]
        return int(idnt), mix_x, mix_a, mix_b, torch.FloatTensor(bpm_x), torch.FloatTensor(bpm_a), torch.FloatTensor(bpm_b), self.condition



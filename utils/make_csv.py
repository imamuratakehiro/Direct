import torch
from torch.utils.data import Dataset
#from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
import soundfile as sf
#from tqdm_table import tqdm_table
#from tqdm import tqdm
import json

from func import file_exist

class MyError(Exception):
    pass

def trackname(no):
    if no in range(1, 10):
        track_name = "Track0000{}".format(no)
    elif no in range(10, 100):
        track_name = "Track000{}".format(no)
    elif no in range(100, 1000):
        track_name = "Track00{}".format(no)
    elif no in range(1000, 2101):
        track_name = "Track0{}".format(no)
    else:
        raise MyError(f"Argument no is not correct ({no}).")
    return track_name

def silence_checker(data):
    max = np.max(data)
    min = np.min(data)
    if -1e-4 < max - min and max - min < 1e-4:
        return True
    else:
        return False

def silence_checker_precise(data, silent_seg_len = 0.25):
    """データの中に一定時間の無音がないセグを判定"""
    len_data = data.shape[0]
    offset = 0
    duration = int(len_data * silent_seg_len)
    while offset + duration <= len_data:
        if silence_checker(data[offset : offset+duration]):
            return True
        offset += 5000
    return False

def songname_csv(dataset="musdb18", type="train"):
    path = f"/nas03/assets/Dataset/MUSDB18/wav/{type}"
    file_list = os.listdir(path)
    print(file_list)
    np.savetxt(f"./aeunet5triplet/metadata/{dataset}/{type}_{dataset}.txt", np.array(file_list), delimiter = ",", fmt = "%s")

def songseg_csv(dataset="musdb18", type="train", s=10):
    path = f"./aeunet5triplet/metadata/{dataset}/{type}_{dataset}.txt"
    filename_list = np.loadtxt(path)
    for name in filename_list:
        filepath_mix   = "/nas03/assets/Dataset/MUSDB18/wav/" + name + "/mixture.wav"
        filepath_drums = "/nas03/assets/Dataset/MUSDB18/wav/" + name + "/drums.wav"
        filepath_bass  = "/nas03/assets/Dataset/MUSDB18/wav/" + name + "/bass.wav"
        filepath_other = "/nas03/assets/Dataset/MUSDB18/wav/" + name + "/other.wav"
        filepath_vocal = "/nas03/assets/Dataset/MUSDB18/wav/" + name + "/vocal.wav"

def silence_csv(dataset="musdb18", type="train", s=10, overrap=0.5, silent_seg_len=0.25):
    #if type=="train":
    #    path = "./unet5/metadata/slakh/train1200_redux.json"
    #elif type=="valid":
    #    path = "./unet5/metadata/slakh/valid_redux.json"
    #elif type=="test":
    #    path = "./unet5/metadata/slakh/test_redux_136.json"
    if dataset == "musdb18":
        path = f"./metadata/{dataset}/{type}_{dataset}.txt"
        filename_list = np.loadtxt(path, delimiter = ",", dtype="unicode")
    elif dataset == "slakh":
        if type == "train":
            path = f"./metadata/{dataset}/train1200_redux.json"
        if type == "test":
            path = f"./metadata/{dataset}/test_redux_136.json"
        if type == "valid":
            path = f"./metadata/{dataset}/valid_redux.json"
        #path = f"./aeunet5triplet/metadata/{dataset}/slakh2100.txt"
        slakh_open = open(path, 'r')
        filename_list = json.load(slakh_open)

    #slakh_open = open(path, 'r')
    #slakh_list = json.load(slakh_open)
    all = []
    mix = []
    stem = []
    if dataset == "musdb18":
        silence_data = [["track_name", "seg", "mix_silence", "drums_silence", "bass_silence", "vocals_silence",  "other_silence"]]
    elif dataset == "slakh":
        silence_data = [["track_name", "seg", "mix_silence", "drums_silence", "bass_silence", "piano_silence", "guitar_silence", "residuals_silence"]]
    #silent_seg_len = 0.25
    #s = 10
    for name in filename_list:
        if dataset == "slakh":
            name = trackname(name)
            filepath_mix       = "/nas03/assets/Dataset/slakh-2100_2/" + name + "/mix.flac"
            filepath_drums     = "/nas03/assets/Dataset/slakh-2100_2/" + name + "/submixes/drums.wav"
            filepath_bass      = "/nas03/assets/Dataset/slakh-2100_2/" + name + "/submixes/bass.wav"
            filepath_guitar    = "/nas03/assets/Dataset/slakh-2100_2/" + name + "/submixes/guitar.wav"
            filepath_piano     = "/nas03/assets/Dataset/slakh-2100_2/" + name + "/submixes/piano.wav"
            filepath_residuals = "/nas03/assets/Dataset/slakh-2100_2/" + name + "/submixes/residuals.wav"
            data_mix,       sr = sf.read(filepath_mix)
            data_drums,     sr = sf.read(filepath_drums)
            data_bass,      sr = sf.read(filepath_bass)
            data_piano,     sr = sf.read(filepath_piano)
            data_guitar,    sr = sf.read(filepath_guitar)
            data_residuals, sr = sf.read(filepath_residuals)
        elif dataset == "musdb18":
            filepath_mix   = f"/nas03/assets/Dataset/MUSDB18/wav/{type}/" + name + "/mixture.wav"
            filepath_drums = f"/nas03/assets/Dataset/MUSDB18/wav/{type}/" + name + "/drums.wav"
            filepath_bass  = f"/nas03/assets/Dataset/MUSDB18/wav/{type}/" + name + "/bass.wav"
            filepath_other = f"/nas03/assets/Dataset/MUSDB18/wav/{type}/" + name + "/other.wav"
            filepath_vocal = f"/nas03/assets/Dataset/MUSDB18/wav/{type}/" + name + "/vocals.wav"
            data_mix,       sr = sf.read(filepath_mix)
            data_drums,     sr = sf.read(filepath_drums)
            data_bass,      sr = sf.read(filepath_bass)
            data_vocal,     sr = sf.read(filepath_vocal)
            data_other,     sr = sf.read(filepath_other)
        #num = data_mix.shape[0] // (sr * s)
        offset = 0
        duration = sr * s
        #overrap = 0.5
        #count = 1
        #print(name, num)
        #for i in range(1, num+1):
        while offset + duration <= data_mix.shape[0]:
            #all.append([name, i])
            seg_mix       = data_mix[offset:offset+duration]
            seg_drums     = data_drums[offset:offset+duration]
            seg_bass      = data_bass[offset:offset+duration]
            if dataset == "musdb18":
                seg_vocal     = data_vocal[offset:offset+duration]
                seg_other     = data_other[offset:offset+duration]
            elif dataset == "slakh":
                seg_piano     = data_piano[offset:offset+duration]
                seg_guitar    = data_guitar[offset:offset+duration]
                seg_residuals = data_residuals[offset:offset+duration]
            if not silence_checker_precise(seg_mix, silent_seg_len=silent_seg_len):
                mix.append([name, offset])
                #print([trackname(name), i])
            if dataset == "musdb18":
                if not(silence_checker_precise(seg_drums, silent_seg_len=silent_seg_len) or silence_checker_precise(seg_bass, silent_seg_len=silent_seg_len)
                        or silence_checker_precise(seg_vocal, silent_seg_len=silent_seg_len) or silence_checker_precise(seg_other, silent_seg_len=silent_seg_len)):
                    stem.append([name, offset])
                    print([name, offset])
                silence_data.append([name,
                                    offset,
                                    silence_checker_precise(seg_mix, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_drums, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_bass, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_vocal, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_other, silent_seg_len=silent_seg_len)])
            elif dataset == "slakh":
                if not(silence_checker_precise(seg_drums, silent_seg_len=silent_seg_len) or silence_checker_precise(seg_bass, silent_seg_len=silent_seg_len)
                        or silence_checker_precise(seg_piano, silent_seg_len=silent_seg_len) or silence_checker_precise(seg_guitar, silent_seg_len=silent_seg_len)
                        or silence_checker_precise(seg_residuals, silent_seg_len=silent_seg_len)):
                    stem.append([name, offset])
                    print([name, offset])
                silence_data.append([name,
                                    offset,
                                    silence_checker_precise(seg_mix, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_drums, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_bass, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_piano, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_guitar, silent_seg_len=silent_seg_len),
                                    silence_checker_precise(seg_residuals, silent_seg_len=silent_seg_len)])
            offset += int(duration * overrap)
            #print([trackname(name), i])
    #np.savetxt(f"./aeunet5triplet/metadata/{dataset}/{s}s/{type}_{dataset}_{s}s.txt",                         np.array(all),          delimiter = ",", fmt = "%s")
    dir_path = f"./metadata/{dataset}/{s}s_no_silence_or{overrap}_{silent_seg_len}"
    file_exist(dir_path)
    np.savetxt(dir_path + f"/{type}_{dataset}_{s}s_mix.txt",          np.array(mix),          delimiter = ",", fmt = "%s")
    np.savetxt(dir_path + f"/{type}_{dataset}_{s}s_stems.txt",        np.array(stem),         delimiter = ",", fmt = "%s")
    np.savetxt(dir_path + f"/{type}_{dataset}_{s}s_silence_data.txt", np.array(silence_data), delimiter = ",", fmt = "%s")

def separate_knn_testdata(dataset="slakh", type="test", s=10):
    reduce_silence_list = ["mix", "stems"]
    for reduce_silence in reduce_silence_list:
        counter = 0
        datafile = np.loadtxt(f"./metadata/{dataset}/{s}s_no_silence_or0.5_0.25/{type}_slakh_{s}s_{reduce_silence}.txt", delimiter = ",", dtype = "unicode")
        filenamelist = []
        test_for_plot = []
        test_for_knn = []
        for data in datafile:
            if not data[0] in filenamelist:
                filenamelist.append(data[0])
            if counter % 5 == 0:
                test_for_knn.append(data)
            else:
                test_for_plot.append(data)
            counter += 1
        dir_path = f"./aeunet5triplet/metadata/{dataset}/knn_test_no_silence_or0.5_0.25"
        file_exist(dir_path)
        np.savetxt(dir_path + f"/{s}s_for_plot.txt",      np.array(test_for_plot),delimiter = ",", fmt = "%s")
        np.savetxt(dir_path + f"/{s}s_for_knn.txt",       np.array(test_for_knn), delimiter = ",", fmt = "%s")
        np.savetxt(dir_path + f"/{s}s_test_filename.txt", np.array(filenamelist), delimiter = ",", fmt = "%s")


if "__main__" == __name__:
    #silence_csv(type="train", s=10)
    #songname_csv(dataset="musdb18", type="train")
    #songname_csv(dataset="musdb18", type="test")
    silence_csv(dataset="slakh",   type="valid", s=10, overrap=0.5, silent_seg_len=0.25)
    #silence_csv(dataset="slakh",   type="test" , s=10, overrap=0.5, silent_seg_len=0.25)
    silence_csv(dataset="slakh",   type="valid", s=5,  overrap=0.5, silent_seg_len=0.25)
    #silence_csv(dataset="slakh",   type="test" , s=5,  overrap=0.5, silent_seg_len=0.25)
    silence_csv(dataset="slakh",   type="valid", s=3,  overrap=0.5, silent_seg_len=0.25)
    #silence_csv(dataset="slakh",   type="test" , s=3,  overrap=0.5, silent_seg_len=0.25)
    #separate_knn_testdata(dataset="slakh", type="test", s=10)
    #separate_knn_testdata(dataset="slakh", type="test", s=5)
    #separate_knn_testdata(dataset="slakh", type="test", s=3)
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
#from torchvision.transforms import ToTensor, Lambda
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torchvision.transforms as Tv
import numpy as np
import os
#import stempeg
import csv
import pandas as pd
import soundfile as sf
#from tqdm_table import tqdm_table
#from tqdm import tqdm
import json
import librosa.core as lc
import librosa
import random

from utils.func import stft, trackname, detrackname
from utils.logger import MyLoggerModel, MyLoggerTrain
from .dataset_triplet import TripletDatasetOneInst, LoadSongWithLabel
from .dataset_zume import (
    CreatePseudo,
    SongDataForPreTrain,
    PsdLoader,
    TestLoader,
    TripletLoaderFromList,
    Condition32Loader,
    TripletLoaderEachTime,
    TripletLoaderAllDiff,
    SongDataForUNetNotPseudo,
    SongDataForUNetPseudo,
    TripletLoaderMix,
    NormalSongsLoader,
    NormalSongLoaderForMOSZume2023,
    TestSongPair,
    TripletLoaderMixForSerialED,
    DataLoaderForABXZume2024
    )


class PreTrainDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.mylogger = MyLoggerModel()
    
    def prepare_data(self) -> None:
        print(f"\n----------------------------------------")
        print(f"Use dataset {self.cfg.datasetname}.")
        print(f"The frame size is setted to {self.cfg.f_size}.")
        print("----------------------------------------")
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.mylogger.s_dataload()
            self.trainset = SongDataForPreTrain(mode="train", cfg=self.cfg)
            psd = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst, psd_mine=False) for inst in self.cfg.inst_list]
            not_psd = [TestLoader(mode="valid", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            psd_mine = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst, psd_mine=True) for inst in self.cfg.inst_list]
            self.validset = psd + not_psd + psd_mine
            self.mylogger.f_dataload()
        if stage == "test" or stage is None:
            #self.testset = [PsdLoader(mode="test", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            psd = [PsdLoader(mode="test", cfg=self.cfg, inst=inst, psd_mine=False) for inst in self.cfg.inst_list]
            not_psd = [TestLoader(mode="test", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            psd_mine = [PsdLoader(mode="test", cfg=self.cfg, inst=inst, psd_mine=True) for inst in self.cfg.inst_list]
            self.testset = psd + not_psd + psd_mine
        if stage == "predict"  or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch_train,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.validset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.validset))]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.testset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.testset))]
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class TripletDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg,
        ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.mylogger = MyLoggerModel()
    
    def prepare_data(self) -> None:
        print(f"\n----------------------------------------")
        print(f"Use dataset {self.cfg.datasetname}.")
        print(f"The frame size is setted to {self.cfg.f_size}.")
        print("----------------------------------------")
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.mylogger.s_dataload()
            # train
            if self.cfg.triplet_not_list:
                if self.cfg.all_diff:
                    # TODO: instの自由度なし
                    self.trainset = TripletLoaderAllDiff(mode="train", cfg=self.cfg, inst=self.cfg.inst)
                else:
                    self.trainset = TripletLoaderEachTime(mode="train", cfg=self.cfg)
            else:
                self.trainset = TripletLoaderFromList(mode="train", cfg=self.cfg)
            # valid
            psd = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst, psd_mine=False) for inst in self.cfg.inst_list]
            not_psd = [TestLoader(mode="valid", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            psd_mine = [PsdLoader(mode="valid", cfg=self.cfg, inst=inst, psd_mine=True) for inst in self.cfg.inst_list]
            abx_2024 = [DataLoaderForABXZume2024(self.cfg, "test", inst) for inst in self.cfg.inst_list]
            self.validset = psd + not_psd + psd_mine + abx_2024
            if self.cfg.triplet_not_list:
                if self.cfg.all_diff:
                    # TODO: instの自由度なし
                    self.validset.insert(0, TripletLoaderAllDiff(mode="valid", cfg=self.cfg, inst=self.cfg.inst))
                else:
                    self.validset.insert(0, TripletLoaderEachTime(mode="valid", cfg=self.cfg))
            else:
                self.validset.insert(0, TripletLoaderFromList(mode="valid", cfg=self.cfg))
            self.mylogger.f_dataload()
        if stage == "test" or stage is None:
            # test
            psd = [PsdLoader(mode="test", cfg=self.cfg, inst=inst, psd_mine=False) for inst in self.cfg.inst_list]
            not_psd = [TestLoader(mode="test", cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            psd_mine = [PsdLoader(mode="test", cfg=self.cfg, inst=inst, psd_mine=True) for inst in self.cfg.inst_list]
            mos_zume = [NormalSongLoaderForMOSZume2023(cfg=self.cfg, inst=inst) for inst in self.cfg.inst_list]
            abx_2024 = [DataLoaderForABXZume2024(self.cfg, "test", inst) for inst in self.cfg.inst_list]
            self.testset = psd + not_psd + psd_mine + mos_zume + abx_2024
            self.testset.insert(0, Condition32Loader(mode="test", cfg=self.cfg))
        if stage == "predict"  or stage is None:
            pass
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.cfg.batch_train,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.validset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.validset))]
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return [DataLoader(
            dataset=self.testset[i],
            batch_size=self.cfg.batch_test,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        ) for i in range(len(self.testset))]
    
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

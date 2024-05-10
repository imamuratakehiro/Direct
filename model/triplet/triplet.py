from typing import Any, Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import ModuleList as MList, ModuleDict as MDict
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import museval

from utils.func import file_exist, knn_psd, tsne_psd, istft, tsne_psd_marker, TorchSTFT, tsne_not_psd, evaluate
from ..csn import ConditionalSimNet1d
from ..tripletnet import CS_Tripletnet


class Triplet(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        cfg,
        ckpt_model_path,
        ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # network
        self.net = net
        model_checkpoint = {}
        if ckpt_model_path is not None:
            print("== Loading pretrained model...")
            checkpoint = torch.load(ckpt_model_path)
            for key in checkpoint["state_dict"]:
                model_checkpoint[key.replace("net.","")] = checkpoint["state_dict"][key]
            self.net.load_state_dict(model_checkpoint)
            print("== pretrained model was loaded!")
        print(net)

        # loss function
        self.loss_unet  = nn.L1Loss(reduction="mean")
        self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="none") #バッチ平均
        self.loss_l2      = nn.MSELoss(reduction="sum")
        self.loss_mrl     = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均
        self.loss_cross_entropy = nn.BCEWithLogitsLoss(reduction="mean")

        self.song_type = ["anchor", "positive", "negative"]
        self.sep_eval_type = ["sdr", "sir", "isr", "sar"]
        self.recorder = MDict({})
        for step in ["Train", "Valid"]:
            self.recorder[step] = MDict({})
            self.recorder[step]["loss_all"] = MeanMetric()
            self.recorder[step]["loss_unet"] = MDict({type: MeanMetric() for type in self.song_type})
            self.recorder[step]["loss_unet"]["all"] = MeanMetric()
            self.recorder[step]["loss_triplet"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
            self.recorder[step]["loss_triplet"]["all"] = MeanMetric()
            self.recorder[step]["loss_recog"] = MeanMetric()
            self.recorder[step]["recog_acc"] = MDict({inst: BinaryAccuracy() for inst in cfg.inst_list})
            self.recorder[step]["dist_p"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
            self.recorder[step]["dist_n"] = MDict({inst: MeanMetric() for inst in cfg.inst_list})
        self.recorder["Test"] = MDict({})
        self.recorder["Test"]["recog_acc"] = MDict({inst: BinaryAccuracy() for inst in cfg.inst_list})
        self.n_sound = {}
        self.recorder_psd = {}
        for step in ["Valid", "Test"]:
            self.recorder_psd[step] = {}
            self.recorder_psd[step]["label"] = {}
            self.recorder_psd[step]["vec"] = {}
            self.recorder_psd[step]["sep"] = {}
            self.n_sound[step] = {}
            for psd in ["psd", "not_psd", "psd_mine", "mos_zume"]:
                self.recorder_psd[step]["label"][psd] = {inst: [] for inst in cfg.inst_list}
                self.recorder_psd[step]["vec"][psd] = {inst: [] for inst in cfg.inst_list}
                self.n_sound[step][psd] = {inst: 0 for inst in cfg.inst_list}
            for s in ["sdr", "sir", "isr", "sar"]:
                self.recorder_psd[step]["sep"][s] = {inst: MeanMetric() for inst in cfg.inst_list}
        self.result_abx = {inst: [] for inst in cfg.inst_list}
        self.stft = TorchSTFT(cfg=cfg)
        self.cfg = cfg
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.recorder["Valid"]["loss_all"].reset()
        for type in self.song_type:
            self.recorder["Valid"]["loss_unet"][type].reset()
        self.recorder["Valid"]["loss_unet"]["all"].reset()
        for inst in self.cfg.inst_list:
            self.recorder["Valid"]["loss_triplet"][inst].reset()
            self.recorder["Valid"]["loss_recog"][inst].reset()
            self.recorder["Valid"]["recog_acc"][inst].reset()
            self.recorder["Valid"]["dist_p"][inst].reset()
            self.recorder["Valid"]["dist_n"][inst].reset()
        self.recorder["Valid"]["loss_triplet"]["all"].reset()
    
    def get_loss_unet(self, X, y, pred_mask):
        batch = X.shape[0]
        loss = 0
        for idx, inst in enumerate(self.cfg.inst_list): #個別音源でロスを計算
            pred = X * pred_mask[inst]
            loss += self.loss_unet(pred, y[:,idx])
        return loss / len(self.cfg.inst_list)
    
    def get_loss_unet_triposi(self, X, y, pred, triposi):
        # triplet positionのところのみ分離ロスを計算
        batch = X.shape[0]
        loss = 0
        for idx, c in enumerate(triposi): #個別音源でロスを計算
            loss += self.loss_unet(pred[self.cfg.inst_list[c.item()]][idx], y[idx, c])
        return loss / batch

    def get_loss_triplet(self, e_a, e_p, e_n, triposi):
        #batch = triposi.shape[0]
        tnet = CS_Tripletnet(ConditionalSimNet1d().to(e_a.device))
        distp, distn = tnet(e_a, e_p, e_n, triposi)
        target = torch.FloatTensor(distp.size()).fill_(1).to(distp.device) # 1で埋める
        loss = self.loss_triplet(distn, distp, target) # トリプレットロス
        loss_all = torch.sum(loss)/len(triposi)
        loss_inst  = {inst: torch.sum(loss[torch.where(triposi==i)])/len(torch.where(triposi==i)[0])  if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_list)}
        dist_p_all = {inst: torch.sum(distp[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_list)}
        dist_n_all = {inst: torch.sum(distn[torch.where(triposi==i)])/len(torch.where(triposi==i)[0]) if len(torch.where(triposi==i)[0]) != 0 else 0 for i,inst in enumerate(self.cfg.inst_list)}
        return loss_all, loss_inst, dist_p_all, dist_n_all
    
    def get_loss_recognise1(self, emb, cases, l = 1e5):
        batch = len(cases)
        loss_emb = 0
        for b in range(batch):
            for c, inst in enumerate(self.inst_list):
                if cases[b][c] == "1":
                    # 0ベクトルとembedded vectorの距離
                    dist_0 = F.pairwise_distance(emb[inst][b], torch.zeros_like(emb[inst][b], device=cases.device), 2)
                    target = torch.ones_like(dist_0).to(cases.device) #1で埋める
                    # 0ベクトルとembedded vectorの距離がmarginよりも大きくなることを期待
                    loss_emb += self.loss_mrl(dist_0, torch.zeros_like(dist_0, device=cases.device), target)
                else:
                    loss_emb += self.loss_l2(emb[inst][b], torch.zeros_like(emb[inst][b], device=cases.device))
        return loss_emb / batch # バッチ平均をとる

    def get_loss_recognise2(self, probability, cases):
        batch = len(cases)
        loss_recognise = 0
        for b in range(batch):
            for c, inst in enumerate(self.cfg.inst_list):
                # 実際の有音無音判定と予想した有音確率でクロスエントロピーロスを計算
                loss_recognise += self.loss_cross_entropy(probability[inst][b], cases[b, c])
        return loss_recognise / batch / len(self.cfg.inst_list)
    
    def transform(self, x_wave, y_wave):
        device = x_wave.device
        if self.cfg.complex:
            x = self.stft.transform(x_wave); y = self.stft.transform(y_wave)
        else:
            x, _ = self.stft.transform(x_wave); y, _ = self.stft.transform(y_wave)
        return x, y

    def clone_for_additional(self, a_x, a_y, p_x, p_y, n_x, n_y, s_a, s_p, s_n, triposi):
        if triposi.dim() == 2: # [b, a]で入ってる
            x_a, x_p, x_n, y_a, y_p, y_n, a_s, p_s, n_s, tp = [], [], [], [], [], [], [], [], [], []
            for i, ba in enumerate(triposi):
                # basic
                x_a.append(a_x[i].clone()); x_p.append(p_x[i].clone()); x_n.append(n_x[i].clone())
                y_a.append(a_y[i].clone()); y_p.append(p_y[i].clone()); y_n.append(n_y[i].clone())
                a_s.append(s_a[i]); p_s.append(s_p[i]); n_s.append(s_n[i]); tp.append(ba[0])
                if not ba[1].item() == -1:
                    # additional
                    x_a.append(a_x[i].clone()); x_p.append(n_x[i].clone()); x_n.append(p_x[i].clone())
                    y_a.append(a_y[i].clone()); y_p.append(n_y[i].clone()); y_n.append(p_y[i].clone())
                    a_s.append(s_a[i]); p_s.append(s_n[i]); n_s.append(s_p[i]); tp.append(ba[1])
            return (torch.stack(x_a, dim=0),
                    torch.stack(y_a, dim=0),
                    torch.stack(x_p, dim=0),
                    torch.stack(y_p, dim=0),
                    torch.stack(x_n, dim=0),
                    torch.stack(y_n, dim=0),
                    torch.stack(a_s, dim=0),
                    torch.stack(p_s, dim=0),
                    torch.stack(n_s, dim=0),
                    torch.stack(tp, dim=0))
        else:
            return a_x, a_y, p_x, p_y, n_x, n_y, s_a, s_p, s_n, triposi

    def forward(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        (a_x_wave, a_y_wave, p_x_wave, p_y_wave, n_x_wave, n_y_wave,
        sound_a, sound_p, sound_n,
        bpm_a, bpm_p, bpm_n, triposi) = batch
        # stft
        with torch.no_grad():
            a_x, a_y = self.transform(a_x_wave, a_y_wave)
            p_x, p_y = self.transform(p_x_wave, p_y_wave)
            n_x, n_y = self.transform(n_x_wave, n_y_wave)
        a_x, a_y, p_x, p_y, n_x, n_y, sound_a, sound_p, sound_n, triposi = self.clone_for_additional(a_x, a_y, p_x, p_y, n_x, n_y, sound_a, sound_p, sound_n, triposi)

        if self.cfg.bpm:
            # TODO:instの決め方テキトーです
            a_e, a_prob = self.net(a_x, bpm_a)
            p_e, p_prob = self.net(p_x, bpm_p)
            n_e, n_prob = self.net(n_x, bpm_n)
        else:
            a_e, a_prob, a_pred = self.net(a_x)
            p_e, p_prob, p_pred = self.net(p_x)
            n_e, n_prob, n_pred = self.net(n_x)
        # get loss
        loss_unet_a = self.get_loss_unet_triposi(a_x, a_y, a_pred, triposi)
        loss_unet_p = self.get_loss_unet_triposi(p_x, p_y, p_pred, triposi)
        loss_unet_n = self.get_loss_unet_triposi(n_x, n_y, n_pred, triposi)
        loss_triplet_all, loss_triplet, dist_p, dist_n = self.get_loss_triplet(a_e, p_e, n_e, triposi)
        loss_recog = self.get_loss_recognise2(a_prob, sound_a) + self.get_loss_recognise2(p_prob, sound_p) + self.get_loss_recognise2(n_prob, sound_n)
        # record loss
        loss_all = (loss_unet_a + loss_unet_p + loss_unet_n)*self.cfg.unet_rate\
                    + loss_triplet_all*self.cfg.triplet_rate\
                    + loss_recog*self.cfg.recog_rate
        loss_unet = {
            "anchor": loss_unet_a.item(),
            "positive": loss_unet_p.item(),
            "negative": loss_unet_n.item(),
            "all": loss_unet_a.item() + loss_unet_p.item() + loss_unet_n.item()
        }
        loss_triplet["all"] = loss_triplet_all.item()
        prob = {inst: torch.concat([a_prob[inst], p_prob[inst], n_prob[inst]], dim=0) for inst in self.cfg.inst_list}
        cases = torch.concat([sound_a, sound_p, sound_n], dim=0)
        return loss_all, loss_unet, loss_triplet, dist_p, dist_n, loss_recog.item(), prob, cases
    
    def model_step(self, mode:str, batch):
        loss_all, loss_unet, loss_triplet, dist_p, dist_n, loss_recog, prob, cases = self.forward(batch)
        # update and log metrics
        self.recorder[mode]["loss_all"](loss_all)
        self.log(f"{mode}/loss_all", self.recorder[mode]["loss_all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # unet
        for type in self.song_type:
            self.recorder[mode]["loss_unet"][type](loss_unet[type])
            self.log(f"{mode}/loss_unet_{type}", self.recorder[mode]["loss_unet"][type], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self.recorder[mode]["loss_unet"]["all"](loss_unet["all"])
        self.log(f"{mode}/loss_unet_all", self.recorder[mode]["loss_unet"]["all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # triplet
        for inst in self.cfg.inst_list:
            self.recorder[mode]["loss_triplet"][inst](loss_triplet[inst])
            self.recorder[mode]["dist_p"][inst](dist_p[inst])
            self.recorder[mode]["dist_n"][inst](dist_n[inst])
            self.log(f"{mode}/loss_triplet_{inst}", self.recorder[mode]["loss_triplet"][inst], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{mode}/dist_p_{inst}",       self.recorder[mode]["dist_p"][inst],       on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            self.log(f"{mode}/dist_n_{inst}",       self.recorder[mode]["dist_n"][inst],       on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        self.recorder[mode]["loss_triplet"]["all"](loss_triplet["all"])
        self.log(f"{mode}/loss_triplet_all", self.recorder[mode]["loss_triplet"]["all"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        # recognize
        self.recorder[mode]["loss_recog"](loss_recog)
        self.log(f"{mode}/loss_recog", self.recorder[mode]["loss_recog"], on_step=True, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        for idx,inst in enumerate(self.cfg.inst_list):
            self.recorder[mode]["recog_acc"][inst](prob[inst], cases[:,idx])
            self.log(f"{mode}/recog_acc_{inst}", self.recorder[mode]["recog_acc"][inst], on_step=True, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
        # return loss or backpropagation will fail
        return loss_all
    
    def model_step_psd(self, mode:str, batch, idx):
        ID, ver, seg, data_wave, c = batch
        with torch.no_grad():
            if self.cfg.complex:
                data = self.stft.transform(data_wave)
            else:
                data, _ = self.stft.transform(data_wave)
        embvec, _, _ = self.net(data)
        if self.cfg.test_valid_norm:
            embvec = torch.nn.functional.normalize(embvec, dim=1)
        csn_valid = ConditionalSimNet1d().to(embvec.device)
        self.recorder_psd[mode]["label"][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
        self.recorder_psd[mode]["vec"][self.cfg.inst_list[idx]].append(csn_valid(embvec, c))
    
    def model_step_knn_tsne(self, mode:str, batch, idx, psd: str):
        ID, ver, seg, data_wave, _, c = batch
        with torch.no_grad():
            if self.cfg.complex:
                data = self.stft.transform(data_wave)
            else:
                data, _ = self.stft.transform(data_wave)
        embvec, _, _ = self.net(data)
        if self.cfg.test_valid_norm:
            embvec = torch.nn.functional.normalize(embvec, dim=1)
        if len(self.cfg.inst_list) == 1:
            self.recorder_psd[mode]["label"][psd][self.cfg.inst].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec"][psd][self.cfg.inst].append(embvec)
        else:
            csn_valid = ConditionalSimNet1d().to(embvec.device)
            self.recorder_psd[mode]["label"][psd][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
            self.recorder_psd[mode]["vec"][psd][self.cfg.inst_list[idx]].append(csn_valid(embvec, c))
    
    def model_step_abx(self, mode: str, batch, idx):
        idnt, mix_x_wave, mix_a_wave, mix_b_wave, _, _, _, c = batch
        with torch.no_grad():
            if self.cfg.complex:
                mix_x = self.stft.transform(mix_x_wave)
                mix_a = self.stft.transform(mix_a_wave)
                mix_b = self.stft.transform(mix_b_wave)
            else:
                mix_x, _ = self.stft.transform(mix_x_wave)
                mix_a, _ = self.stft.transform(mix_a_wave)
                mix_b, _ = self.stft.transform(mix_b_wave)
        emb_x, _, _ = self.net(mix_x)
        emb_a, _, _ = self.net(mix_a)
        emb_b, _, _ = self.net(mix_b)
        csn = ConditionalSimNet1d().to(emb_x.device)
        emb_x = csn(emb_x, c)
        emb_a = csn(emb_a, c)
        emb_b = csn(emb_b, c)
        if self.cfg.test_valid_norm:
            emb_x = torch.nn.functional.normalize(emb_x, dim=1)
            emb_a = torch.nn.functional.normalize(emb_a, dim=1)
            emb_b = torch.nn.functional.normalize(emb_b, dim=1)
        dist_XA = torch.norm(emb_x - emb_a, dim=1, keepdim=True)
        dist_XB = torch.norm(emb_x - emb_b, dim=1, keepdim=True)
        idnt = torch.unsqueeze(idnt, dim=-1)
        self.result_abx[self.cfg.inst_list[idx]].append(torch.concat([idnt, dist_XA, dist_XB], dim=1))

    def training_step(
        self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_all = self.model_step("Train", batch)
        return loss_all

    def print_loss(self, mode:str):
        # unet
        print("\n\n== U-Net Loss ==")
        loss_unet = {type: self.recorder[mode]["loss_unet"][type].compute() for type in self.song_type}
        print(f"{mode} average loss UNet (anchor, positive, negative)  : {loss_unet['anchor']:2f}, {loss_unet['positive']:2f}, {loss_unet['negative']:2f}")
        loss_unet_all = self.recorder[mode]["loss_unet"]["all"].compute()
        print(f"{mode} average loss UNet            (all)              : {loss_unet_all:2f}")
        # triplet
        print("\n== Triplet Loss ==")
        for inst in self.cfg.inst_list:
            loss_triplet = self.recorder[mode]["loss_triplet"][inst].compute()
            dist_p = self.recorder[mode]["dist_p"][inst].compute()
            dist_n = self.recorder[mode]["dist_n"][inst].compute()
            print(f"{mode} average loss {inst:9}(Triplet, dist_p, dist_n) : {loss_triplet:2f}, {dist_p:2f}, {dist_n:2f}")
        loss_triplet_all = self.recorder[mode]["loss_triplet"]["all"].compute()
        print(f"{mode} average loss all      (Triplet)                 : {loss_triplet_all:2f}")
        # recognize
        print("\n== Recognize ==")
        print(f"{mode} average loss Recognize     : {self.recorder[mode]['loss_recog'].compute():2f}")
        for inst in self.cfg.inst_list:
            recog_acc = self.recorder[mode]["recog_acc"][inst].compute()
            print(f"{mode} average accuracy {inst:9} : {recog_acc*100:2f} %")
        print(f"\n== {mode} average loss all : {self.recorder[mode]['loss_all'].compute():2f}\n")

    def output_label_vec(self, mode, epoch, label, vec, path):
        lv = np.concatenate([label, vec], axis=1)
        dirpath = self.cfg.output_dir + path
        file_exist(dirpath)
        pd.DataFrame(lv).to_csv(dirpath + f"/normal_{mode}_e={epoch}.csv", header=False, index=False)
    
    def knn_tsne(self, mode:str, psd:str):
        print(f"\n== {psd} ==")
        acc_all = 0
        for inst in self.cfg.inst_list:
            label = torch.concat(self.recorder_psd[mode]["label"][psd][inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.recorder_psd[mode]["vec"][psd][inst], dim=0).to("cpu").numpy()
            acc = knn_psd(label, vec, self.cfg, psd=False if psd == "not_psd" else True) # knn
            self.log(f"{mode}/knn_{psd}_{inst}", acc, on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=False)
            if psd == "psd_mine":
                tsne_psd_marker(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            elif psd == "psd":
                tsne_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            elif psd == "not_psd":
                self.output_label_vec(mode=mode, epoch=self.current_epoch, label=label, vec=vec, path=f"/csv/{psd}/{inst}")
                tsne_not_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            print(f"{mode} knn accuracy {inst:<10} {psd:<8} : {acc*100}%")
            acc_all += acc
        self.log(f"{mode}/knn_{psd}_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        print(f"\n{mode} knn accuracy average {psd:<8}   : {acc_all/len(self.cfg.inst_list)*100}%")
        self.recorder_psd[mode]["label"][psd] = {inst:[] for inst in self.cfg.inst_list}; self.recorder_psd[mode]["vec"][psd] = {inst:[] for inst in self.cfg.inst_list}
    
    def output_label_vec_for_mos(self):
        mode = "Test"; psd = "mos_zume"
        for inst in self.cfg.inst_list:
            label = torch.concat(self.recorder_psd[mode]["label"][psd][inst], dim=0).to("cpu").numpy()
            vec   = torch.concat(self.recorder_psd[mode]["vec"][psd][inst], dim=0).to("cpu").numpy()
            self.output_label_vec(mode=mode, epoch=self.current_epoch, label=label, vec=vec, path=f"/csv/{psd}/{inst}")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.print_loss("Train")

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx == 0:
            loss_all = self.model_step("Valid", batch)
        elif dataloader_idx > 0 and dataloader_idx < n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx-1, psd="psd")
        elif dataloader_idx >= n_inst + 1 and dataloader_idx < 2*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst + 1 and dataloader_idx < 3*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - 2*n_inst, psd="psd_mine")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.print_loss("Valid")
        self.knn_tsne("Valid", psd="psd")
        self.knn_tsne("Valid", psd="not_psd")
        self.knn_tsne("Valid", psd="psd_mine")
    
    def evaluate_separated(self, reference, estimate):
        # assume mix as estimates
        B_r, S_r, T_r = reference.shape
        B_e, S_e, T_e = estimate.shape
        reference = torch.reshape(reference, (B_r, T_r, S_r))
        estimate  = torch.reshape(estimate, (B_e, T_e, S_e))
        if T_r > T_e:
            reference = reference[:, :T_e]
        scores = {}
        scores = {"sdr":0, "isr":0, "sir":0, "sar":0}
        # Evaluate using museval
        score = museval.evaluate(references=reference.to("cpu"), estimates=estimate.to("cpu"))
        for i,key in enumerate(list(scores.keys())):
            #print(score[i].shape)
            scores[key] = np.mean(score[i])
        return scores

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx == 0:
            """cases_x, cases_y, cases = batch
            I, B, C, T = cases_y.shape
            #print(I, B, C, T)
            if self.cfg.complex:
                cases_x = self.stft.transform(cases_x)
                phase = None
            else:
                cases_x, phase = self.stft.transform(cases_x)
            c_e, c_prob, c_pred = self.net(cases_x)
            if not self.cfg.mel:
                for idx,c in enumerate(cases):
                    #and self.n_sound < 5):
                    #sound = self.stft.detransform(cases_x[idx], phase[idx], param[0,idx], param[1,idx])
                    # TODO: complex = TrueだとphaseがNoneでidxがないって怒られるからそこを直す！いっそモデルの中で波にしちゃうのあり？てかそのロス追加する？
                    if self.cfg.complex:
                        sound = self.stft.detransform(cases_x[idx])
                    else:
                        sound = self.stft.detransform(cases_x[idx], phase[idx])
                    #sound = self.stft.detransform(cases_x[idx], phase[idx])
                    path = self.cfg.output_dir+f"/sound/mix"
                    file_exist(path)
                    #soundfile.write(path + f"/separate{self.n_sound}_mix.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                    for j,inst in enumerate(self.cfg.inst_list):
                        if c[j] == 1:
                            #sound = self.stft.detransform(cases_x[idx]*c_mask[inst][idx], phase[idx], param[0,idx], param[1,idx])
                            if self.cfg.complex:
                                sound = self.stft.detransform(c_pred[inst][idx])
                            else:
                                sound = self.stft.detransform(c_pred[inst][idx], phase[idx])
                            scores = self.evaluate_separated(cases_y[idx, j:j+1], torch.unsqueeze(sound, dim=1))
                            for s in self.sep_eval_type:
                                self.recorder_psd["Test"]["sep"][s][inst](scores[s])
                            path = self.cfg.output_dir+f"/sound/{inst}"
                            file_exist(path)
                            #soundfile.write(path + f"/separate{self.n_sound}_{inst}.wav", np.squeeze(sound.to("cpu").numpy()), self.cfg.sr)
                    #self.n_sound += 1
            for idx,inst in enumerate(self.cfg.inst_list):
                self.recorder["Test"]["recog_acc"][inst](c_prob[inst], cases[:,idx])"""
            pass

        elif dataloader_idx > 0 and dataloader_idx < n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx-1, psd="psd")
        elif dataloader_idx >= n_inst + 1 and dataloader_idx < 2*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst + 1 and dataloader_idx < 3*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - 2*n_inst, psd="psd_mine")
        elif dataloader_idx >= 3*n_inst + 1 and dataloader_idx < 4*n_inst + 1:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 1 - 3*n_inst, psd="mos_zume")
        elif dataloader_idx >= 4*n_inst + 1 and dataloader_idx < 5*n_inst + 1:
            self.model_step_abx("Test", batch, dataloader_idx - 1 - 4*n_inst)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        print()
        for inst in self.cfg.inst_list:
            recog_acc = self.recorder["Test"]["recog_acc"][inst].compute()
            print(f"Test average accuracy {inst:9} : {recog_acc*100: 2f} %")
        for inst in self.cfg.inst_list:
            for s in self.sep_eval_type:
                print(f"{s} {inst:9}: {self.recorder_psd['Test']['sep'][s][inst].compute()}")
        # abx
        for inst in self.cfg.inst_list:
            result_abx = torch.concat(self.result_abx[inst], dim = 0).to("cpu").numpy()
            file_exist(self.cfg.output_dir + f"/csv/abx2024/{inst}")
            pd.DataFrame(result_abx, columns=["identifier", "dist_XA", "dist_XB"]).to_csv(self.cfg.output_dir + f"/csv/abx2024/{inst}/result.csv", index=False)
        self.output_label_vec_for_mos()
        self.knn_tsne("Test", psd="psd")
        self.knn_tsne("Test", psd="not_psd")
        self.knn_tsne("Test", psd="psd_mine")

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss_all",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
        #return torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.lr)


if __name__ == "__main__":
    _ = Triplet(None, None, None, None)
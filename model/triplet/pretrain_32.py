from typing import Any, Dict, Tuple

import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric

from utils.func import file_exist, knn_psd, tsne_psd, tsne_psd_marker, TorchSTFT, tsne_not_psd
from ..csn import ConditionalSimNet1d


class PreTrain32(LightningModule):
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
        cfg) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        assert cfg.condition32, f"Please change the value condition32 to True. Now, it is {cfg.condition32}."

        # network
        self.net = net
        print(net)

        # loss function
        self.loss_triplet = nn.MarginRankingLoss(margin=cfg.margin, reduction="mean") #バッチ平均
        self.loss_mse = nn.MSELoss(reduction="mean")

        # for averaging loss across batches
        self.train_loss_mix  = MeanMetric()
        self.train_loss_inst = MeanMetric()
        
        # for valid and test
        self.recorder_psd = {}
        self.n_sound = {}
        for step in ["Valid", "Test"]:
            self.recorder_psd[step] = {}
            self.recorder_psd[step]["label"] = {}
            self.recorder_psd[step]["vec"] = {}
            self.n_sound[step] = {}
            for psd in ["psd", "not_psd", "psd_mine"]:
                self.recorder_psd[step]["label"][psd] = {inst: [] for inst in cfg.inst_list}
                self.recorder_psd[step]["vec"][psd] = {inst: [] for inst in cfg.inst_list}
                self.n_sound[step][psd] = {inst: 0 for inst in cfg.inst_list}
        self.stft = TorchSTFT(cfg=cfg)
        self.cfg = cfg
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass
    
    def forward_mix(self, X):
        # 順伝播
        emb_mix, _ = self.net(X)
        return emb_mix
    
    def forward_inst(self, y):
        emb_insts = []
        for idx,inst in enumerate(self.cfg.inst_list):
            emb_inst, _, _ = self.net(y[:,idx])
            emb_insts.append(emb_inst)
        return torch.stack(emb_insts, dim=1)

    def model_step(self, batch):
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        mix, condition, emb_target = batch
        if self.cfg.complex:
            mix = self.stft.transform(mix)
        else:
            mix, _ = self.stft.transform(mix)
        emb_mix   = self.forward_mix(mix)
        # loss
        loss_mix = self.loss_mse(emb_mix, emb_target)
        return loss_mix

    def model_step_knn_tsne(self, mode:str, batch, idx, psd: str):
        ID, ver, seg, data_wave, c = batch
        if self.cfg.complex:
            mix = self.stft.transform(data_wave)
        else:
            mix, _ = self.stft.transform(data_wave)
        embvec = self.forward_mix(mix)
        if self.cfg.test_valid_norm:
            embvec = torch.nn.functional.normalize(embvec, dim=1)
        csn_valid = ConditionalSimNet1d().to(embvec.device)
        self.recorder_psd[mode]["label"][psd][self.cfg.inst_list[idx]].append(torch.stack([ID, ver], dim=1))
        self.recorder_psd[mode]["vec"][psd][self.cfg.inst_list[idx]].append(csn_valid(embvec, c))
    
    def knn_tsne(self, mode:str, psd: str):
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
                tsne_not_psd(label, vec, mode, self.cfg, dir_path=self.cfg.output_dir+f"/figure/{inst}/{mode}_e={self.current_epoch}/{psd}", current_epoch=self.current_epoch) # tsne
            print(f"{mode} knn accuracy {inst:<10} {psd:<8} : {acc*100}%")
            acc_all += acc
        self.log(f"{mode}/knn_{psd}_avr", acc_all/len(self.cfg.inst_list), on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
        print(f"\n{mode} knn accuracy average {psd:<8}   : {acc_all/len(self.cfg.inst_list)*100}%")
        self.recorder_psd[mode]["label"][psd] = {inst:[] for inst in self.cfg.inst_list}; self.recorder_psd[mode]["vec"][psd] = {inst:[] for inst in self.cfg.inst_list}

    def training_step(
        self, batch, batch_idx: int
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss_mix = self.model_step(batch)
        # update and log metrics
        self.train_loss_mix(loss_mix)
        self.log("train/loss", self.train_loss_mix, on_step=True, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss_mix

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        print(f"\nTrain average loss <input 32condition > : {self.train_loss_mix.compute()}")

    def validation_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx >= 0 and dataloader_idx < n_inst:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx, psd="psd")
        elif dataloader_idx >= n_inst and dataloader_idx < 2*n_inst:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst and dataloader_idx < 3*n_inst:
            self.model_step_knn_tsne("Valid", batch, dataloader_idx - 2*n_inst, psd="psd_mine")

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.knn_tsne("Valid", psd="psd")
        self.knn_tsne("Valid", psd="not_psd")
        self.knn_tsne("Valid", psd="psd_mine")

    def test_step(self, batch, batch_idx: int, dataloader_idx=0) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        n_inst = len(self.cfg.inst_list)
        if dataloader_idx >= 0 and dataloader_idx < n_inst:
            self.model_step_knn_tsne("Test", batch, dataloader_idx, psd="psd")
        elif dataloader_idx >= n_inst and dataloader_idx < 2*n_inst:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - n_inst, psd="not_psd")
        elif dataloader_idx >= 2*n_inst and dataloader_idx < 3*n_inst:
            self.model_step_knn_tsne("Test", batch, dataloader_idx - 2*n_inst, psd="psd_mine")

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
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
                    "monitor": self.cfg.monitor_sch,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
        
        #return torch.optim.Adam(self.trainer.model.parameters(), lr=self.cfg.lr)


if __name__ == "__main__":
    _ = PreTrain32(None, None, None, None)
from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchmetrics
from lightning.pytorch.accelerators import find_usable_cuda_devices
from asteroid.data import DNSDataset
from DCCRN import DCCRN

from framework import ABF, MultiResolutionSTFTLoss, SPKDLoss, build_review_kd, ReviewKD
import feature_extraction
from params import params
import config as cfg

CUDA_LAUNCH_BLOCKING=1

class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, teacher, student, RLF, feature_fusion, sftf_loss, spkd_loss, cfg):
        super().__init__()
        self.save_hyperparameters()

        #load teacher (pre-trained)
        self.teacher = teacher
        self.teacher_weight = torch.load(cfg.teacher_weight_path)
        self.teacher.load_state_dict(self.teacher_weight)
        #freeze teacher
        for paras in self.teacher.parameters():
            paras.requires_grad = False

        #load student model
        self.student = student

        #SPKD loss
        self.spkd_loss = spkd_loss
        #featuer fusion
        self.abf = feature_fusion

        #MR SFTF loss
        self.base_loss = sftf_loss[1]
        self.student_review_kd = build_review_kd(self.student)
        self.kd_loss_weight = cfg.kd_loss_weight


    def forward(self, x):
        return self.rlf(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.cuda(), y.cuda()

        losses = {"kd_loss": 0, "base_loss": 0}
        
        feature_maps = {'encoder':0,'decoder':0,'CLSTM_real':0,'CLSTM_img':0}
        for ft_map in feature_maps:
            # getting student and teacher features
            student_features, student_preds = self.student_review_kd(X, ft_map)

            teacher_features = feature_extraction.FE_DCCRN(teacher).extract_feature_maps(X, ft_map)

            # calculating review kd loss
            for sf, tf in zip(student_features, teacher_features):
                losses['kd_loss'] += self.spkd_loss(sf, tf,'batchmean')

            # calculating base_loss (Multi-resolution STFT)
            losses['base_loss'] = self.base_loss(student_preds, y)

            loss = losses['kd_loss'] * self.kd_loss_weight
            loss += losses['base_loss']

            self.log('train_loss', loss)
            for key in losses:
                self.log(f'train_{key}', losses[key])
            teacher_features.clear()
            #save loss:
            feature_maps[ft_map] = loss
        loss = sum(feature_maps.values())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # calculate running average of accuracy
        x, y = batch[0:2]
        _, student_preds = self.rlf(x)
        student_preds = torch.max(student_preds.data, 1)[1]
        acc = torchmetrics.functional.accuracy(student_preds, y)

        self.log("val_acc", acc)
        return acc
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.rlf.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4,
        )
        return optimizer
    
    def train_dataloader(self):
        train_dataset = DNSDataset("/root/NTH_student/asteroid/egs/dns_challenge/baseline/data")
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=cfg.batch,  # max 3696 * snr types
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last=True,
                                                sampler=None)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = DNSDataset("/root/NTH_student/asteroid/egs/dns_challenge/baseline/data")
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=cfg.batch, 
                                                shuffle=False, 
                                                num_workers=4)
        return val_loader


#setup
torch.set_float32_matmul_precision('high')

# initialize models
teacher = DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                  kernel_num=cfg.kernel_num)
student = DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                  kernel_num=cfg.kernel_num)

# initialize trainer
trainer = pl.Trainer(max_epochs=5, accelerator="gpu", devices=[1])

# initialize knowledge distillation module
kd_module = KnowledgeDistillation(teacher, 
                                  student, 
                                  RLF=ReviewKD, 
                                  feature_fusion=ABF, 
                                  sftf_loss=MultiResolutionSTFTLoss, 
                                  spkd_loss=SPKDLoss,
                                  cfg=cfg)

# train the student network using knowledge distillation
trainer.fit(kd_module)



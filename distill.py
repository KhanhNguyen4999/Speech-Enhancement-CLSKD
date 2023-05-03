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

from framework import RLF, ABF, MultiResolutionSTFTLoss, SPKDLoss
from params import params
import config as cfg

batch_size = 32
CUDA_LAUNCH_BLOCKING=1

class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, teacher, student, RLF, feature_fusion, sftf_loss, spkd_loss, params):
        super().__init__()
        #load teacher (pre-trained)
        self.teacher = teacher


        self.student = student
        self.spkd_loss = spkd_loss
        self.abf = feature_fusion
        #MR SFTF loss
        self.base_loss = sftf_loss[1]
        self.rlf = RLF(self.student, abf_to_use = self.abf)
        self.kd_loss_weight = 0.6


    
    def forward(self, x):
        return self.rlf(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.cuda(), y.cuda()

        losses = {"kd_loss": 0, "base_loss": 0}

        # getting student and teacher features
        student_features, student_preds = self.rlf(X)
        #need encoder-decoder extracion
        teacher_features, teacher_preds = self.teacher(X,)

        teacher_features = teacher_features[1:]

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
            lr=0.1,
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
                                  RLF=RLF, 
                                  feature_fusion=ABF, 
                                  sftf_loss=MultiResolutionSTFTLoss, 
                                  spkd_loss=SPKDLoss,
                                  params=params)

# train the student network using knowledge distillation
trainer.fit(kd_module)



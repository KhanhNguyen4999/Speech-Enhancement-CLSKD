from typing import Any
import numpy as np
import os
import argparse
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pytorch_lightning as pl
from lightning.pytorch.accelerators import find_usable_cuda_devices
from asteroid.data import DNSDataset
from asteroid.models import DCCRNet, DCCRNet_mini
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from DCCRN import DCCRN
import yaml
from pprint import pprint
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
from dataloader import create_dataloader
from tools_for_model import cal_pesq, cal_stoi
from torch_stoi import NegSTOILoss


from framework import MultiResolutionSTFTLoss, SPKDLoss, build_review_kd
import feature_extraction
import config as cfg

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1"

class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, teacher, student, sftf_loss, spkd_loss, cfg):
        super().__init__()
        self.automatic_optimization = True

        #load teacher (pre-trained)
        self.teacher = teacher
        #self.teacher_checkpoint = torch.load(cfg.teacher_weight_path)
        #self.teacher.load_state_dict(self.teacher_checkpoint['model'])

        #freeze teacher
        for paras in self.teacher.parameters():
            paras.requires_grad = False

        #load student model
        self.student = student

        #SPKD loss
        self.spkd_loss = spkd_loss
    
        #base loss - MRSFTF loss
        self.stft_loss = sftf_loss(fft_sizes=[cfg.fft_len], win_lengths=[cfg.win_len],hop_sizes=[cfg.win_inc])

        #student_loss:
        self.student_loss = nn.CrossEntropyLoss()
        #val_loss function
        self.sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        self.stoi = PITLossWrapper(NegSTOILoss(sample_rate=16000), pit_from='pw_pt')

    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        X, y,_ = batch

        # calculating based-loss (Multi-resolution STFT)
        student_preds = self.student(X)
        #base_loss = self.stft_loss(student_preds.squeeze(),y)[1]
        with torch.no_grad():
            teacher_preds = self.teacher(X)
        
        #student_loss = self.student_loss(student_preds,y.to(torch.long))
        base_loss = self.stft_loss(student_preds.squeeze(),y)[1]
        spkd_loss_func = self.spkd_loss(student_preds, teacher_preds, reduction='batchmean')
        spkd_loss = spkd_loss_func()

        loss = base_loss + spkd_loss
        # logging training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # calculate running average of accuracy
        x, y,_ = batch
        student_preds = self.student(x)


        estimated_wavs = student_preds.cpu().detach().squeeze().numpy()
        clean_wavs = y.cpu().detach().numpy()

        #pesq-stoi
        pesq = cal_pesq(estimated_wavs, clean_wavs)  ## 98
        stoi = cal_stoi(estimated_wavs, clean_wavs)

        # reshape for sum
        pesq = np.reshape(pesq, (1, -1))
        stoi = np.reshape(stoi, (1, -1))

        avg_pesq = sum(pesq[0]) / len(x)
        avg_stoi = sum(stoi[0]) / len(x)

        #si_sdr
        si_sdr = self.sisdr(student_preds.detach(),y.detach().unsqueeze(1))

        #stoi
        stoi = self.stoi(student_preds.detach(),y.detach().unsqueeze(1))
        # logging val_pesq, val_stoi
        self.log_dict({"val_pesq": avg_pesq, "val_stoi": avg_stoi, "si_sdr":si_sdr,"stoi":stoi}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
        return {"val_pesq": avg_pesq, "val_stoi": avg_stoi, "si_sdr":si_sdr,"stoi":stoi}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
        return optimizer
    
    def train_dataloader(self):
        train_dataset = DNSDataset("/root/NTH_student/train_loader")
        train_loader = create_dataloader(mode='train',dataset=train_dataset)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = DNSDataset("/root/NTH_student/test_loader")
        val_loader = create_dataloader(mode='valid',dataset=val_dataset)
        return val_loader



# setup float type
torch.set_float32_matmul_precision('high')

# read config file
parser = argparse.ArgumentParser()
with open("./Speech_Enhancement_new/knowledge_distillation_CLSKD/conf.yml") as f:
    def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
conf, plain_args = parse_args_as_dict(parser, return_plain_args=True)
pprint(conf)

# initialize models
teacher =  DCCRNet.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
student =  DCCRNet_mini(
        **conf["filterbank"], **conf["masknet"], sample_rate=conf["data"]["sample_rate"])



# initalize checkpoint
checkpoint_callback = ModelCheckpoint(
                    dirpath='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD/checkpoint',
                    filename='model-{epoch:02d}-{si_sdr:.2f}-{stoi:.2f}',
                    save_top_k=2,
                    monitor='si_sdr',
                    mode='min',
                    verbose=True)


# initialize trainer
trainer = pl.Trainer(max_epochs=cfg.max_epochs, 
                    accelerator="gpu", 
                    devices=[1],
                    default_root_dir='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD',
                    callbacks=[checkpoint_callback]
                    )

# initialize knowledge distillation module
kd_module = KnowledgeDistillation(teacher, 
                                student,                                              
                                sftf_loss=MultiResolutionSTFTLoss, 
                                spkd_loss=SPKDLoss,
                                cfg=cfg)

# train the student network using knowledge distillation
trainer.fit(kd_module)

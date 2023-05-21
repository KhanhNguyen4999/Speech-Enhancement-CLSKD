from typing import Any
import numpy as np
import os 
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchmetrics
from lightning.pytorch.accelerators import find_usable_cuda_devices
from asteroid.data import DNSDataset
from DCCRN import DCCRN
from dataloader import create_dataloader
from tools_for_model import cal_pesq, cal_stoi


from framework import MultiResolutionSTFTLoss, SPKDLoss, build_review_kd
import feature_extraction
import config as cfg

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1"
CUDA_LAUNCH_BLOCKING=1

class KnowledgeDistillation(pl.LightningModule):
    def __init__(self, teacher, student, sftf_loss, spkd_loss, cfg):
        super().__init__()
        self.automatic_optimization = True

        #load teacher (pre-trained)
        self.teacher = teacher
        self.teacher_checkpoint = torch.load(cfg.teacher_weight_path)
        self.teacher.load_state_dict(self.teacher_checkpoint['model'])

        #freeze teacher
        for paras in self.teacher.parameters():
            paras.requires_grad = False

        #load student model
        self.student = student

        #SPKD loss
        self.spkd_loss = spkd_loss
    
        #base loss - MRSFTF loss
        self.stft_loss = sftf_loss(fft_sizes=[cfg.fft_len], win_lengths=[cfg.win_len],hop_sizes=[cfg.win_inc])

    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        X, y,_ = batch

        # getting student features
        student_extraction = feature_extraction.DCCRN(self.student)
        student_features = student_extraction.extract_feature_maps(X)
        student_encoder,student_decoder, student_clstm_real,student_clstm_img = (student_features[0],
                                                                                 student_features[2],
                                                                                 student_features[1][0],
                                                                                 student_features[1][1])
        
        model_encoder = build_review_kd(student_encoder,'encoder')
        student_features_encoder = model_encoder(X)
        
        model_decoder = build_review_kd(student_decoder,'decoder')
        student_features_decoder = model_decoder(X)

        #getting teacher features
        teacher_extraction = feature_extraction.DCCRN(self.teacher)
        teacher_features = teacher_extraction.extract_feature_maps(X)
        teacher_encoder, teacher_decoder, teacher_clstm_real, teacher_clstm_img = (teacher_features[0], 
                                                                                   teacher_features[2], 
                                                                                   teacher_features[1][0], 
                                                                                   teacher_features[1][1])

        
        # calculating based-loss (Multi-resolution STFT)
        student_preds = self.student(X, is_feat=True)
        base_loss = self.stft_loss(student_preds,y)[1]
        

        feature_maps_loss = {'encoder':0,'decoder':0,'clstm_real':0,'clstm_img':0}

        ############## ENCODER loss ######################
        losses = {"kd_loss": 0}
        # calculating review kd loss
        for sf in student_features_encoder:
            tf = teacher_encoder
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            losses['kd_loss'] += kd_loss()
        loss = losses['kd_loss']
        #save loss for each feature map:
        feature_maps_loss['encoder'] = loss


        ############## DECODER loss ######################
        losses = {"kd_loss": 0}
        # calculating review kd loss
        for sf in student_features_decoder:
            tf = teacher_decoder
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            losses['kd_loss'] += kd_loss()
        loss = losses['kd_loss']
        #save loss for each feature map:
        feature_maps_loss['decoder'] = loss



        ############## C-LSTM REAL loss ######################
        losses = {"kd_loss": 0}
        # calculating review kd loss
        
        kd_loss = self.spkd_loss(student_clstm_real, teacher_clstm_real,reduction=None)
        losses['kd_loss'] = kd_loss()
        loss = losses['kd_loss']
        #save loss for each feature map:
        feature_maps_loss['clstm_real'] = loss


        ############## C-LSTM IMAGE loss ######################
        losses = {"kd_loss": 0}
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_img, teacher_clstm_img,reduction=None)
        losses['kd_loss'] = kd_loss()
        loss = losses['kd_loss']
        #save loss for each feature map:
        feature_maps_loss['clstm_img'] = loss
        

        ############### Calculating all losses [Encoder + Decoder + C-Lstm_real + C-Lstm_img] ##################
        loss = base_loss + sum(feature_maps_loss.values())

        # logging
        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # clear memory
        teacher_extraction.remove_hook()
        student_extraction.remove_hook()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # calculate running average of accuracy
        x, y,_ = batch
        student_preds = self.student(x, is_feat=True)
        estimated_wavs = student_preds.cpu().detach().numpy()
        clean_wavs = y.cpu().detach().numpy()

        pesq = cal_pesq(estimated_wavs, clean_wavs)  ## 98
        stoi = cal_stoi(estimated_wavs, clean_wavs)

        # reshape for sum
        pesq = np.reshape(pesq, (1, -1))
        stoi = np.reshape(stoi, (1, -1))

        avg_pesq = sum(pesq[0]) / len(x)
        avg_stoi = sum(stoi[0]) / len(x)

        self.log_dict({"val_pesq": avg_pesq, "val_stoi": avg_stoi}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
        #monitor memory
        return {"val_pesq": avg_pesq, "val_stoi": avg_stoi}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=cfg.lr, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(
        #     self.student.parameters(),
        #     lr=cfg.lr,
        #     nesterov=True,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        return optimizer
    
    def train_dataloader(self):
        train_dataset = DNSDataset("/root/NTH_student/train_loader")
        # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
        #                                         batch_size=cfg.batch,  # max 3696 * snr types
        #                                         shuffle=True,
        #                                         num_workers=0,
        #                                         pin_memory=True,
        #                                         drop_last=True,
        #                                         sampler=None)
        train_loader = create_dataloader(mode='train',dataset=train_dataset)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = DNSDataset("/root/NTH_student/test_loader")
        # val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
        #                                         batch_size=cfg.batch, 
        #                                         shuffle=False, 
        #                                         num_workers=0)
        val_loader = create_dataloader(mode='valid',dataset=val_dataset)
        return val_loader


def main(): 
    #setup
    torch.set_float32_matmul_precision('high')

    # initialize models
    teacher =  DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                    kernel_num=cfg.kernel_num)
    student =  DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                    kernel_num=cfg.kernel_num)

    # initialize trainer
    trainer = pl.Trainer(max_epochs=cfg.batch, 
                        accelerator="gpu", 
                        devices=[1],
                        log_every_n_steps=5,
                        default_root_dir='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD')

    # initialize knowledge distillation module
    kd_module = KnowledgeDistillation(teacher, 
                                    student,                                              
                                    sftf_loss=MultiResolutionSTFTLoss, 
                                    spkd_loss=SPKDLoss,
                                    cfg=cfg)

    # train the student network using knowledge distillation
    trainer.fit(kd_module)


if __name__== "__main__":
    main()
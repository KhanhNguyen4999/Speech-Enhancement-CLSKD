from typing import Any
import numpy as np
import os 
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pytorch_lightning as pl
from lightning.pytorch.accelerators import find_usable_cuda_devices
from asteroid.data import DNSDataset
from DCCRN import DCCRN
from dataloader import create_dataloader
from tools_for_model import cal_pesq, cal_stoi


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
        student_encoder,student_decoder, student_clstm_real,student_clstm_img = (student_features["encoder"],
                                                                                 student_features["decoder"],
                                                                                 student_features["clstm"][0][0],
                                                                                 student_features["clstm"][0][1])
        
        model_encoder = build_review_kd(student_encoder,'encoder')
        student_features_encoder = model_encoder(X)
        
        model_decoder = build_review_kd(student_decoder,'decoder')
        student_features_decoder = model_decoder(X)

        # getting teacher features
        teacher_extraction = feature_extraction.DCCRN(self.teacher)
        teacher_features = teacher_extraction.extract_feature_maps(X)
        teacher_encoder, teacher_decoder, teacher_clstm_real, teacher_clstm_img = (teacher_features["encoder"], 
                                                                                   teacher_features["decoder"], 
                                                                                   teacher_features["clstm"][0][0], 
                                                                                   teacher_features["clstm"][0][1])

        
        # calculating based-loss (Multi-resolution STFT)
        teacher_preds = self.teacher(X, is_feat=True)
        base_loss = self.stft_loss(teacher_preds,y)[1]
        

        feature_maps_loss = {'encoder':0,'decoder':0,'clstm_real':0,'clstm_img':0}

        ############## ENCODER loss ######################
        losses = {"kd_loss": 0}
        # calculating review kd loss
        for sf, tf in zip(student_features_encoder,teacher_encoder):
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            losses['kd_loss'] += kd_loss()
        loss = losses['kd_loss']
        #save loss for each feature map:
        feature_maps_loss['encoder'] = loss


        ############## DECODER loss ######################
        losses = {"kd_loss": 0}
        # calculating review kd loss
        for sf, tf in zip(student_features_decoder,teacher_decoder):
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            losses['kd_loss'] += kd_loss()
        loss = losses['kd_loss']
        #save loss for each feature map:
        feature_maps_loss['decoder'] = loss


        ############## C-LSTM REAL loss ######################
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_real, teacher_clstm_real,reduction=None)
        feature_maps_loss['clstm_real'] = kd_loss()
        

        ############## C-LSTM IMAGE loss ######################
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_img, teacher_clstm_img,reduction=None)
        feature_maps_loss['clstm_img'] = kd_loss()
        

        ############### Calculating all losses [Encoder + Decoder + C-Lstm_real + C-Lstm_img] ##################
        loss = base_loss + sum(feature_maps_loss.values())

        # logging training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # clear memory for register hook
        teacher_extraction.remove_hook()
        student_extraction.remove_hook()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # calculate running average of accuracy
        x, y,_ = batch
        _, _, real_spec, img_spec, student_preds = self.student(x)
        val_loss = self.student.loss(student_preds, y.float(), real_spec, img_spec)

        estimated_wavs = student_preds.cpu().detach().numpy()
        clean_wavs = y.cpu().detach().numpy()

        pesq = cal_pesq(estimated_wavs, clean_wavs)  ## 98
        stoi = cal_stoi(estimated_wavs, clean_wavs)

        # reshape for sum
        pesq = np.reshape(pesq, (1, -1))
        stoi = np.reshape(stoi, (1, -1))

        avg_pesq = sum(pesq[0]) / len(x)
        avg_stoi = sum(stoi[0]) / len(x)

        self.log_dict({"val_loss":val_loss, "val_pesq": avg_pesq, "val_stoi": avg_stoi}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
        #monitor memory
        return {"val_loss":val_loss, "val_pesq": avg_pesq, "val_stoi": avg_stoi}
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=cfg.learning_rate, weight_decay=5e-4)
        # optimizer = torch.optim.SGD(
        #     self.student.parameters(),
        #     lr=cfg.learning_rate,
        #     nesterov=True,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        return optimizer
    
    def train_dataloader(self):
        train_dataset = DNSDataset("/root/NTH_student/train_loader")
        train_loader = create_dataloader(mode='train',dataset=train_dataset)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = DNSDataset("/root/NTH_student/test_loader")
        val_loader = create_dataloader(mode='valid',dataset=val_dataset)
        return val_loader



#setup
torch.set_float32_matmul_precision('high')

# initialize models
teacher =  DCCRN(rnn_units=cfg.rnn_units, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                kernel_num=cfg.kernel_num)
student =  DCCRN(rnn_units=cfg.rnn_units_student, masking_mode=cfg.masking_mode, use_clstm=cfg.use_clstm,
                kernel_num=cfg.kernel_num_student)

# initalize checkpoint
# checkpoint_callback = ModelCheckpoint(
#                     dirpath='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD/checkpoint',
#                     filename='model-{epoch:02d}-{val_loss:.2f}',
#                     save_top_k=10,
#                     monitor='val_loss')


# initialize trainer
trainer = pl.Trainer(max_epochs=cfg.max_epochs, 
                    accelerator="gpu", 
                    devices=[2],
                    default_root_dir='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD',
                    #callbacks=[checkpoint_callback]
                    )

# initialize knowledge distillation module
kd_module = KnowledgeDistillation(teacher, 
                                student,                                              
                                sftf_loss=MultiResolutionSTFTLoss, 
                                spkd_loss=SPKDLoss,
                                cfg=cfg)

# train the student network using knowledge distillation
trainer.fit(kd_module)

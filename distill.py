from typing import Any
import numpy as np
import pandas as pd
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
from asteroid.data import DNSDataset, LibriMix
from asteroid.models import DCCRNet, DCCRNet_mini
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.metrics import MockWERTracker

import yaml
from pprint import pprint
from asteroid.utils import prepare_parser_from_dict, parse_args_as_dict
from asteroid.metrics import get_metrics
from dataloader import create_dataloader
from asteroid.utils import tensors_to_device
from asteroid.dsp.normalization import normalize_estimates
from tools_for_model import cal_pesq, cal_stoi
from torch_stoi import NegSTOILoss


from framework import MultiResolutionSTFTLoss, SPKDLoss, build_review_kd
import feature_extraction
import config as cfg

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "1"
COMPUTE_METRICS = ["stoi"]


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
        self.stft_loss = sftf_loss(fft_sizes=[512], win_lengths=[32],hop_sizes=[16])
        #self.stft_loss = sftf_loss(fft_sizes=[cfg.fft_len], win_lengths=[cfg.win_len],hop_sizes=[cfg.win_inc])

        #
        self.mixture_path = None

        #val_loss function
        # self.sisdr = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        # self.stoi = PITLossWrapper(NegSTOILoss(sample_rate=16000), pit_from='pw_pt')

    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        X, y = batch
       
        # getting teacher features
        teacher_extraction = feature_extraction.DCCRNet(self.teacher)
        teacher_features = teacher_extraction.extract_feature_maps(X)
        teacher_encoder, teacher_decoder, teacher_clstm_real, teacher_clstm_img = (teacher_features["encoder"], 
                                                                                   teacher_features["decoder"], 
                                                                                   teacher_features["clstm_real"][0], 
                                                                                   teacher_features["clstm_img"][0])

        # getting student features
        student_extraction = feature_extraction.DCCRNet(self.student)
        student_features = student_extraction.extract_feature_maps(X)
        student_encoder,student_decoder, student_clstm_real,student_clstm_img = (student_features["encoder"],
                                                                                 student_features["decoder"],
                                                                                 student_features["clstm_real"][0],
                                                                                 student_features["clstm_img"][0])
        
        # feature fusion (review kd) for student's encoder and decoder
        model_encoder = build_review_kd(student_encoder,'encoder')
        student_features_encoder = model_encoder(X)
        
        model_decoder = build_review_kd(student_decoder,'decoder')
        student_features_decoder = model_decoder(X)

        
        # calculating based-loss (Multi-resolution STFT)
        student_preds = self.student(X)
        base_loss = self.stft_loss(student_preds.squeeze(),y.squeeze())[1]
        

        feature_maps_loss = {'encoder':0.0,'decoder':0.0,'clstm_real':0.0,'clstm_img':0.0}

        ############## ENCODER loss ######################
        loss = 0.0
        # calculating review kd loss
        for sf, tf in zip(student_features_encoder,teacher_encoder):
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            loss += kd_loss()
        #save loss for each feature map:
        feature_maps_loss['encoder'] = loss


        ############## DECODER loss ######################
        loss = 0.0
        # calculating review kd loss
        for sf, tf in zip(student_features_decoder,teacher_decoder):
            kd_loss = self.spkd_loss(sf, tf,'batchmean')
            loss += kd_loss()
        #save loss for each feature map:
        feature_maps_loss['decoder'] = loss


        ############## C-LSTM REAL loss ######################
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_real, teacher_clstm_real,reduction='batchmean')
        feature_maps_loss['clstm_real'] = kd_loss()
        

        ############## C-LSTM IMAGE loss ######################
        # calculating review kd loss
        kd_loss = self.spkd_loss(student_clstm_img, teacher_clstm_img,reduction='batchmean')
        feature_maps_loss['clstm_img'] = kd_loss()
        

        ############## All losses: baseloss + [Encoder + Decoder + C-Lstm_real + C-Lstm_img] ##################
        loss = base_loss + sum(feature_maps_loss.values())

        # logging training loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # clear memory for registration hook
        teacher_extraction.remove_hook()
        student_extraction.remove_hook()

        return loss
    
    def validation_step(self, batch, batch_idx):
        # calculate running average of accuracy
        x, y = batch
        loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
        wer_tracker = (MockWERTracker())
        model_device = next(self.student.parameters()).device
        #print(self.val_dataset.mixture_path)
        series_list = []
        
        for idx in range(len(x)):
            mix = x[idx]
            sources = y[idx]
            mix, sources = tensors_to_device([mix, sources], device=model_device)
            est_sources = self.student(mix.unsqueeze(0))
            loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
            mix_np = mix.cpu().data.numpy()
            sources_np = sources.cpu().data.numpy()
            est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
            # For each utterance, we get a dictionary with the mixture path,
            # the input and output metrics
            utt_metrics = get_metrics(
                mix_np,
                sources_np,
                est_sources_np,
                sample_rate=16000)
            utt_metrics["mix_path"] = self.val_dataset.mixture_path
            est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
            utt_metrics.update(
                **wer_tracker(
                    mix=mix_np,
                    clean=sources_np,
                    estimate=est_sources_np_normalized,
                    sample_rate=16000,
                )
            )
            series_list.append(pd.Series(utt_metrics))

        all_metrics_df = pd.DataFrame(series_list)
        # Print and save summary metrics
        final_results = {}
        for metric_name in COMPUTE_METRICS:
            input_metric_name = "input_" + metric_name
            ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
            final_results[metric_name] = all_metrics_df[metric_name].mean()
            final_results[metric_name + "_imp"] = ldf.mean()

        # print("Overall metrics :")
        # print(final_results)

        # logging metrics
        self.log_dict(final_results, on_step=True, on_epoch=True, prog_bar=True, logger=True)
       
    def configure_optimizers(self):
        optimizer = optim.Adam(self.student.parameters(), lr=cfg.learning_rate)
        return optimizer
    
    def train_dataloader(self):
        # train_dataset = DNSDataset("/root/NTH_student/train_loader")
        train_dataset = LibriMix(
            csv_dir='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD/data/wav16k/min/train-360',
            task='enh_single',
            sample_rate=16000,
            n_src=1,
            segment=3,
        )
        train_loader = create_dataloader(mode='train',dataset=train_dataset)
        return train_loader
    
    def val_dataloader(self):
        # val_dataset = DNSDataset("/root/NTH_student/test_loader")
        val_dataset = LibriMix(
            csv_dir='/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD/data/wav16k/min/dev',
            task='enh_single',
            sample_rate=16000,
            n_src=1,
            segment=3,
        )
        self.val_dataset = val_dataset
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
                    filename='model-{epoch:02d}-{stoi:.4f}',
                    save_top_k=3,
                    monitor='stoi',
                    mode='max',
                    verbose=True)


# initialize trainer
trainer = pl.Trainer(max_epochs=cfg.max_epochs, 
                    accelerator="gpu", 
                    devices=[0],
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

# load best student model 
state_dict = torch.load(checkpoint_callback.best_model_path)
only_student_state_dict = {}
for key,value in state_dict['state_dict'].items():
    if key.startswith('student.'):
        only_student_state_dict[key.replace('student.','')] = value
    else:
        continue
state_dict['state_dict'] = only_student_state_dict

student.load_state_dict(state_dict=state_dict["state_dict"])
student.cpu()

# save the best student model  
to_save = student.serialize()
torch.save(to_save,os.path.join('/root/NTH_student/Speech_Enhancement_new/knowledge_distillation_CLSKD/checkpoint', "the_best_model.pth"))





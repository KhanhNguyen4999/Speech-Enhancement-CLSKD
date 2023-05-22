"""
Configuration for program
"""

#distillation
teacher = 'DCCRN'
student = 'DCCRN'
dataset = 'dns_challenge'
teacher_weight_path = f'/root/NTH_student/Speech_Enhancement_new/DCCRN-with-various-loss-functions/job/dccrn_20230515/chkpt_100.pt'
lr_decay_steps = [12, 17]
lr_decay_rate = 0.1,
weight_decay = 5e-4,

########################### TEACHER ###########################
# model
mode = 'DCCRN'  # DCUNET / DCCRN
info = 'MODEL INFORMATION : IT IS USED FOR FILE NAME'

test = True

# model information
fs = 16000
win_len = 400
win_inc = 100
ola_ratio = win_inc / win_len
fft_len = 512
sam_sec = fft_len / fs
frm_samp = fs * (fft_len / fs)
window_type = 'hamming'

rnn_layers = 2
rnn_units = 256
masking_mode = 'E'
use_clstm = True
kernel_num = [32, 64, 128, 256, 256, 256]  # DCCRN
#kernel_num = [72, 72, 144, 144, 144, 160, 160, 180]  # DCUNET
loss_mode = 'SDR+PMSQE'

# hyperparameters for model train
max_epochs = 20
learning_rate = 0.0005
batch = 64


########################### STUDENT ###########################
rnn_layers_student = 2
rnn_units_student = 64
kernel_num_student = [8, 16, 32, 64, 64, 64]  # DCCRN
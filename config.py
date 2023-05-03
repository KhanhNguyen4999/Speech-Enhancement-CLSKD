"""
Configuration for program
"""

#distillation
teacher = 'DCCRN'
student = 'DCCRN'
dataset = 'dns_challenge'
teacher_weight_path = f'./pretrained/chkpt_1.pt'
lr = 0.1
lr_decay_steps = [12, 17]
lr_decay_rate = 0.1,
weight_decay = 5e-4,
kd_loss_weight = 0.6,


# model
mode = 'DCCRN'  # DCUNET / DCCRN
info = 'MODEL INFORMATION : IT IS USED FOR FILE NAME'

test = True

# path
job_dir = './job/'
logs_dir = './logs/'
chkpt_path = None
# chkpt_model = 'FILE NAME THAT YOU WANT TO LOAD'
# chkpt_path = job_dir + chkpt_model + 'chkpt_88.pt'

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
max_epochs = 100
learning_rate = 0.0005
batch = 8

# Distilling Knowledge via Knowledge Review
# |----> Uses Residual Learning Framework
#        |----> Uses SPKD
#        |----> Uses Attention Based Fusion Module
# |----> Multi-resolution STFT loss (MRSTFT)
# |----> 

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import normalize
import DCCRN
import config as cfg
import feature_extraction

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

########## STFT Loss ##########

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


########## Multi-Resolution STFT Loss ##########

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss

########## Similarity-Preserving Knowledge Distillation ##########

class SPKDLoss(nn.Module):
    def __init__(self, student_output, teacher_output, reduction, **kwargs):
        super().__init__()
        self.student_outputs = student_output
        self.teacher_outputs = teacher_output
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, *args, **kwargs):
        # teacher_outputs = teacher_io_dict[self.teacher_output_path]['output']
        # student_outputs = student_io_dict[self.student_output_path]['output']
        batch_size = self.teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(self.teacher_outputs, self.student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss


########## Attention Based Fusion Module ##########
class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

        # Move the model to the GPU
        self.cuda()

        # Convert the weights to torch.cuda.FloatTensor
        self.conv1[0].weight = nn.Parameter(self.conv1[0].weight.cuda())
        self.conv2[0].weight = nn.Parameter(self.conv2[0].weight.cuda())
        # self.conv1[0].weight = nn.Parameter(self.conv1[0].weight)
        # self.conv2[0].weight = nn.Parameter(self.conv2[0].weight)

    def forward(self, x, y=None, shape=None, out_shape=None, feature_type=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            if feature_type == "encoder":
                y = F.interpolate(y, (shape,w), mode="nearest")
            elif feature_type == "decoder":
                y = F.interpolate(y, (shape,w), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, w), mode="nearest")
        y = self.conv2(x)
        return y, x

class ReviewKD(nn.Module):
    def __init__(
        self, in_channels, out_channels, shapes, out_shapes, feature_maps, ft_type
    ):  
        super(ReviewKD, self).__init__()
        self.shapes = shapes
        self.out_shapes = shapes if out_shapes is None else out_shapes
        self.feature_maps = feature_maps
        self.ft_type = ft_type

        abfs = nn.ModuleList()

        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
        self.abfs = abfs[::-1]
        

    def forward(self, x):
        if self.ft_type == 'encoder':
            x = self.feature_maps[::-1]
            results = []
            out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0], feature_type=self.ft_type)
            results.append(out_features)
            for feature, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
                out_features, res_features = abf(feature, res_features, shape, out_shape,feature_type=self.ft_type)
                results.insert(0, out_features)

        elif self.ft_type == 'decoder':
            x = self.feature_maps
            results = []
            out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0], feature_type=self.ft_type)
            results.append(out_features)
            for feature, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
                    out_features, res_features = abf(feature, res_features, shape, out_shape, feature_type=self.ft_type)
                    results.append(out_features)

        return results
    

def build_review_kd(feature_maps, ft_type):
    if ft_type == 'encoder':
        in_channels = [8, 16, 32, 64, 64, 64]
        #out_channels = [32, 64, 128, 256, 256, 256]
        out_channels = [32, 64, 128, 256, 256, 256]
        shapes = [4,8,16,32,64,128]
        out_shapes = [4,8,16,32,64,128]

    elif ft_type == 'decoder':
        in_channels = [8, 16, 32, 64, 64, 64]
        # out_channels = [2, 32, 64, 128, 256, 256]
        out_channels = [32, 64, 128, 256, 256, 256]
        shapes = [4,8,16,32,64,128]
        # out_shapes = [8,16,32,64,128,256]
        out_shapes = [4,8,16,32,64,128]


    model = ReviewKD(in_channels, out_channels, shapes, out_shapes, feature_maps, ft_type)
    return model


def hcl(fstudent, fteacher, t_type):
    loss_all = 0.0
    for fs, ft in zip(fstudent, fteacher):
        if t_type == 'lstm':
            h,w = fs.shape
        else: n,h,w = fs.shape
        loss = F.mse_loss(fs, ft, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for l in [4,2,1]:
            if l >=h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l,l))
            tmpft = F.adaptive_avg_pool2d(ft, (l,l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


# class ABF(nn.Module):
#     def __init__(self, in_channel, mid_channel, out_channel, fuse):
#         super(ABF, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
#             nn.BatchNorm2d(mid_channel),
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(mid_channel, out_channel,kernel_size=3,stride=1,padding=1,bias=False),
#             nn.BatchNorm2d(out_channel),
#         )
#         if fuse:
#             self.att_conv = nn.Sequential(
#                     nn.Conv2d(mid_channel*2, 2, kernel_size=1),
#                     nn.Sigmoid(),
#                 )
#         else:
#             self.att_conv = None
#         nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
#         nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

#     def forward(self, x, y=None, shape=None, out_shape=None):
#         n,_,h,w = x.shape
#         # transform student features
#         x = self.conv1(x)
#         if self.att_conv is not None:
#             # upsample residual features
#             y = F.interpolate(y, (shape,shape-1), mode="nearest")
#             # fusion
#             z = torch.cat([x, y], dim=1)
#             z = self.att_conv(z)
#             x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
#         # output 
#         if x.shape[-1] != out_shape:
#             x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
#         y = self.conv2(x)
#         return y, x

# class ReviewKD(nn.Module):
#     def __init__(
#         self, student, in_channels, out_channels, shapes, out_shapes,
#     ):  
#         super(ReviewKD, self).__init__()
#         self.student = student
#         self.shapes = shapes
#         self.out_shapes = shapes if out_shapes is None else out_shapes

#         abfs = nn.ModuleList()

#         mid_channel = min(512, in_channels[-1])
#         for idx, in_channel in enumerate(in_channels):
#             abfs.append(ABF(in_channel, mid_channel, out_channels[idx], idx < len(in_channels)-1))
#         self.abfs = abfs[::-1]
        

#     def forward(self, x):
#         student_features = self.student(x,is_feat=True)
#         logit = student_features[1]
#         x = student_features[0][::-1]
#         results = []
#         out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
#         results.append(out_features)
#         for features, abf, shape, out_shape in zip(x[1:], self.abfs[1:], self.shapes[1:], self.out_shapes[1:]):
#             out_features, res_features = abf(features, res_features, shape, out_shape)
#             results.insert(0, out_features)

#         return results, logit
    

# def build_review_kd(student):
#     in_channels = [32, 64, 128, 256, 256, 256]
#     out_channels = [32, 64, 128, 256, 256, 256]
#     mid_channel = 256
    
#     model = ReviewKD(student, in_channels, out_channels, mid_channel)
#     return model
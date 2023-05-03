

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
    def __init__(self, student_output_path, teacher_output_path, reduction, **kwargs):
        super().__init__()
        self.student_output_path = student_output_path
        self.teacher_output_path = teacher_output_path
        self.reduction = reduction

    def matmul_and_normalize(self, z):
        z = torch.flatten(z, 1)
        return normalize(torch.matmul(z, torch.t(z)), 1)

    def compute_spkd_loss(self, teacher_outputs, student_outputs):
        g_t = self.matmul_and_normalize(teacher_outputs)
        g_s = self.matmul_and_normalize(student_outputs)
        return torch.norm(g_t - g_s) ** 2

    def forward(self, student_io_dict, teacher_io_dict, *args, **kwargs):
        teacher_outputs = teacher_io_dict[self.teacher_output_path]['output']
        student_outputs = student_io_dict[self.student_output_path]['output']
        batch_size = teacher_outputs.shape[0]
        spkd_losses = self.compute_spkd_loss(teacher_outputs, student_outputs)
        spkd_loss = spkd_losses.sum()
        return spkd_loss / (batch_size ** 2) if self.reduction == 'batchmean' else spkd_loss


########## Attention Based Fusion Module ##########

# Paper states that the output from the ABF module (single output as
# presented in the ABF flow diagram, fig. 3(a)) is the one of the inputs to
# the next ABF module.

# But the authors' code implementation provides two different outputs, one that
# proceeds to the next ABF module (`residual_output`) and one that
# is the output of the ABF module and which is involved in the loss
# function (`abf_output`)
# The `residual_output` differs from the `abf_output` in terms of the number
# of channels. The `residual_output` has `mid_channels` while the `abf_output`
# has `out_channels`

# In this implementation, we have taken the latter approach

# The second approach can be found in experimental/abf_experiments.py

class ABF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ABF, self).__init__()

        self.mid_channel = 64

        self.conv_to_mid_channel = nn.Sequential(
            nn.Conv2d(in_channel, self.mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mid_channel),
        )
        nn.init.kaiming_uniform_(self.conv_to_mid_channel[0].weight, a=1)

        self.conv_to_out_channel = nn.Sequential(
            nn.Conv2d(self.mid_channel, out_channel, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        nn.init.kaiming_uniform_(self.conv_to_out_channel[0].weight, a=1)

        self.conv_to_att_maps = nn.Sequential(
            nn.Conv2d(self.mid_channel * 2, 2, kernel_size=1),
            nn.Sigmoid(),
        )
        nn.init.kaiming_uniform_(self.conv_to_att_maps[0].weight, a=1)

    def forward(self, student_feature, prev_abf_output, teacher_shape):
        n, c, h, w = student_feature.shape
        student_feature = self.conv_to_mid_channel(student_feature)

        if prev_abf_output is None:
            residual_output = student_feature
        else:
            prev_abf_output = F.interpolate(prev_abf_output, size=(
                teacher_shape, teacher_shape), mode='nearest')

            concat_features = torch.cat(
                [student_feature, prev_abf_output], dim=1)
            attention_maps = self.conv_to_att_maps(concat_features)
            attention_map1 = attention_maps[:, 0].view(n, 1, h, w)
            attention_map2 = attention_maps[:, 1].view(n, 1, h, w)

            residual_output = student_feature * attention_map1 \
                + prev_abf_output * attention_map2

        # the output of the abf is obtained after the residual
        # output is convolved to have `out_channels` channels
        abf_output = self.conv_to_out_channel(residual_output)

        return abf_output, residual_output


########## Residual Learning Framework ##########

class RLF(nn.Module):
    def __init__(self, student, abf_to_use):
        super(RLF, self).__init__()

        self.student = student

        in_channels = [32, 64, 128, 256, 256, 256]
        out_channels = [32, 64, 128, 256, 256, 256]
        # in_channels = [16, 32, 64, 64]
        # out_channels = [16, 32, 64, 64]

        self.shapes = [1, 8, 16, 32, 32]

        ABFs = nn.ModuleList()

        for idx, in_channel in enumerate(in_channels):
            ABFs.append(abf_to_use(in_channel, out_channels[idx]))

        self.ABFs = ABFs[::-1]

    def forward(self, x):
        student_features = self.student(x, is_feat=True)

        student_preds = student_features[1]
        student_features = student_features[0][::-1]

        results = []

        abf_output, residual_output = self.ABFs[0](
            student_features[0], None, self.shapes[0])

        results.append(abf_output)

        for features, abf, shape in zip(student_features[1:], self.ABFs[1:], self.shapes[1:]):
            # here we use a recursive technique to obtain all the ABF
            # outputs and store them in a list
            abf_output, residual_output = abf(features, residual_output, shape)
            results.insert(0, abf_output)

        return results, student_preds

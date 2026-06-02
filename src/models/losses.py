import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class SpectrumLoss(nn.Module):
    def __init__(self):
        super(SpectrumLoss, self).__init__()

    def forward(self, output, target):
        out_fft = torch.fft.rfft(output, dim=-1)
        target_fft = torch.fft.rfft(target, dim=-1)
        return F.mse_loss(torch.abs(out_fft), torch.abs(target_fft))

class PINNStyleTransferLoss(nn.Module):
    def __init__(self):
        super(PINNStyleTransferLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.spectrum_loss = SpectrumLoss()

    def forward(self, output, content_train, style_train):
        loss_recon = self.mse_loss(output, content_train)
        loss_spec = self.spectrum_loss(output, style_train)
        return loss_recon + 0.5 * loss_spec
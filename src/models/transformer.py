'''
Contains Classical NN Architctures as base models
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

class AdaSTNetClassifier(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(AdaSTNetClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, signal, return_features=False):
        c = self.cnn(signal) 
        c = c.permute(0, 2, 1)
        lstm_out, _ = self.lstm(c) 
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) 
        features = torch.sum(attn_weights * lstm_out, dim=1) 
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

class FTClassifier(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(FTClassifier, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal, return_features=False):
        fft_signal = torch.fft.rfft(signal, dim=-1)
        freq_domain_x = torch.abs(fft_signal)
        out = self.cnn(freq_domain_x)
        features = out.view(out.shape[0], -1) 
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits

class LSTMClassifier(nn.Module):
    def __init__(self, input_channels=4, hidden_size=128, num_layers=2, num_classes=4):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, signal, return_features=False):
        x = signal.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        features = lstm_out[:, -1, :] 
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits
    

# ---------------------------------------------------------
# 1. FNO-1D (Fourier Neural Operator)
# The current state-of-the-art for physics and signal data.
# Learns mappings entirely in the frequency domain by mixing modes.
# ---------------------------------------------------------
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes, dtype=torch.cfloat))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, device=x.device, dtype=torch.cfloat)
        # Multiply relevant Fourier modes
        out_ft[:, :, :self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1DClassifier(nn.Module):
    def __init__(self, channels=4, num_classes=4, modes=16, width=64):
        super(FNO1DClassifier, self).__init__()
        self.p = nn.Linear(channels, width)
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, signal):
        x = signal.permute(0, 2, 1) # (batch, length, channels)
        x = self.p(x).permute(0, 2, 1) # (batch, width, length)
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        return self.fc(x)
    

class STFT2DClassifier(nn.Module):
    def __init__(self, channels=4, num_classes=4, n_fft=64):
        super(STFT2DClassifier, self).__init__()
        self.n_fft = n_fft
        
        # 2D CNN acting on the Spectrogram
        self.conv2d = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # Preserve a 4x4 spatial/temporal grid
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.Dropout(0.3), # Added dropout to prevent overfitting on the two dominant classes
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, signal):
        # signal shape: (batch, channels, length)
        batch, ch, length = signal.shape
        
        # Flatten batch and channels to apply STFT
        signal_flat = signal.view(-1, length) 
        
        # Calculate Short-Time Fourier Transform
        # return_complex=True is required in modern PyTorch
        stft_out = torch.stft(signal_flat, n_fft=self.n_fft, return_complex=True)
        
        # Get magnitude spectrogram
        spectrogram = torch.abs(stft_out)
        
        # Reshape back to (batch, channels, freq_bins, time_steps)
        # This acts exactly like an image with 'ch' color channels
        spectrogram = spectrogram.view(batch, ch, spectrogram.size(1), spectrogram.size(2))
        
        # Pass through 2D CNN
        x = self.conv2d(spectrogram)
        return self.fc(x)


# class STFT2DClassifier(nn.Module):
#     def __init__(self, channels=4, num_classes=7, n_fft=64, hop_length=32):
#         super(STFT2DClassifier, self).__init__()
#         self.n_fft = n_fft
#         self.hop_length = hop_length
        
#         # Register the Hann window so it moves to GPU automatically
#         self.register_buffer('window', torch.hann_window(n_fft))
        
#         # SpecAugment (Data Augmentation inside the network)
#         self.freq_masking = T.FrequencyMasking(freq_mask_param=10)
#         self.time_masking = T.TimeMasking(time_mask_param=10)
        
#         # 2D CNN Feature Extractor (Deeper, with Spatial Dropout)
#         self.features = nn.Sequential(
#             nn.Conv2d(channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.GELU(),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.15),
            
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.GELU(),
#             nn.MaxPool2d(2),
#             nn.Dropout2d(0.15),
            
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.GELU(),
#             nn.AdaptiveAvgPool2d((4, 4)), 
#             nn.Dropout2d(0.2)
#         )
        
#         # Classifier
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 4 * 4, 256),
#             nn.BatchNorm1d(256),
#             nn.GELU(),
#             nn.Dropout(0.5), # Heavy dropout before final output
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, signal):
#         batch, ch, length = signal.shape
#         signal_flat = signal.view(-1, length) 
        
#         # 1. Windowed STFT
#         stft_out = torch.stft(
#             signal_flat, 
#             n_fft=self.n_fft, 
#             hop_length=self.hop_length,
#             window=self.window, 
#             return_complex=True
#         )
        
#         # 2. Magnitude & Log Scaling (Crucial for signal processing)
#         spectrogram = torch.abs(stft_out)
#         log_spectrogram = torch.log(spectrogram + 1e-8)
#         log_spectrogram = log_spectrogram.view(batch, ch, log_spectrogram.size(1), log_spectrogram.size(2))
        
#         # 3. Apply SpecAugment ONLY during training to prevent memorization
#         if self.training:
#             log_spectrogram = self.freq_masking(log_spectrogram)
#             log_spectrogram = self.time_masking(log_spectrogram)
        
#         # 4. Pass through network
#         x = self.features(log_spectrogram)
#         return self.classifier(x)
    

    
class DeepFNO1DClassifier32(nn.Module):
    def __init__(self, channels=4, num_classes=4, modes=32, width=32): # Increased modes from 16 to 32
        super(DeepFNO1DClassifier32, self).__init__()
        self.p = nn.Linear(channels, width)
        
        # Two FNO blocks to learn hierarchical frequency features
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        
        self.fc = nn.Sequential(
            # Using MaxPool instead of AdaptiveAvgPool preserves sharp, sudden frequency spikes
            nn.MaxPool1d(4), 
            nn.Flatten(),
            nn.Linear(width * (1000 // 4), 128), # Update '1000' to match your actual signal length
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, signal):
        x = signal.permute(0, 2, 1)
        x = self.p(x).permute(0, 2, 1) 
        
        # Block 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        # Block 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        return self.fc(x)

class DeepFNO1DClassifier64(nn.Module):
    def __init__(self, channels=4, num_classes=4, modes=64, width=64): # Increased modes from 16 to 32
        super(DeepFNO1DClassifier64, self).__init__()
        self.p = nn.Linear(channels, width)
        
        # Two FNO blocks to learn hierarchical frequency features
        self.conv0 = SpectralConv1d(width, width, modes)
        self.w0 = nn.Conv1d(width, width, 1)
        
        self.conv1 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        
        self.fc = nn.Sequential(
            # Using MaxPool instead of AdaptiveAvgPool preserves sharp, sudden frequency spikes
            nn.MaxPool1d(4), 
            nn.Flatten(),
            nn.Linear(width * (1000 // 4), 128), # Update '1000' to match your actual signal length
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, signal):
        x = signal.permute(0, 2, 1)
        x = self.p(x).permute(0, 2, 1) 
        
        # Block 1
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        # Block 2
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        return self.fc(x)

# ---------------------------------------------------------
# 2. FNet-1D (Fourier Token Mixer)
# Replaces computationally heavy Self-Attention with a parameter-free FFT.
# ---------------------------------------------------------
class FNet1DClassifier(nn.Module):
    def __init__(self, channels=4, num_classes=4, hidden_dim=64):
        super(FNet1DClassifier, self).__init__()
        self.proj = nn.Conv1d(channels, hidden_dim, kernel_size=1)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, signal):
        x = self.proj(signal).permute(0, 2, 1) # (batch, seq, dim)
        # FNet Token Mixing (FFT on seq and hidden dims)
        x_fft = torch.fft.fftn(x, dim=(1, 2)).real
        x = self.norm1(x + x_fft)
        # Feed forward
        x = self.norm2(x + self.ffn(x))
        x = x.mean(dim=1) # Global average pooling
        return self.classifier(x)

# ---------------------------------------------------------
# 3. Phase-Magnitude Dual Network
# Processes the structural peaks (magnitude) and timing (phase) separately.
# ---------------------------------------------------------
class PhaseMagnitudeNet(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(PhaseMagnitudeNet, self).__init__()
        self.mag_branch = nn.Sequential(
            nn.Conv1d(channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.phase_branch = nn.Sequential(
            nn.Conv1d(channels, 32, 5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal):
        fft_x = torch.fft.rfft(signal, dim=-1)
        mag = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        m_out = self.mag_branch(mag).flatten(1)
        p_out = self.phase_branch(phase).flatten(1)
        return self.fc(torch.cat([m_out, p_out], dim=1))

# ---------------------------------------------------------
# 4. Multi-Scale Spectral Fusion (MS-SFF)
# Merges raw temporal features with frequency domain features.
# ---------------------------------------------------------
class DualStreamSpectralNet(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(DualStreamSpectralNet, self).__init__()
        self.time_cnn = nn.Sequential(nn.Conv1d(channels, 32, 5, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.freq_cnn = nn.Sequential(nn.Conv1d(channels, 32, 5, stride=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal):
        t_feat = self.time_cnn(signal).flatten(1)
        f_feat = self.freq_cnn(torch.abs(torch.fft.rfft(signal, dim=-1))).flatten(1)
        return self.fc(torch.cat([t_feat, f_feat], dim=1))

# ---------------------------------------------------------
# 5. STFT Spectrogram CNN
# Converts 1D sequence into 2D Time-Frequency map for 2D convolutions.
# ---------------------------------------------------------
class STFTSpectrogramCNN(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(STFTSpectrogramCNN, self).__init__()
        self.cnn2d = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(16, num_classes)

    def forward(self, signal):
        batch, ch, length = signal.shape
        # STFT requires flat input, process and reshape back
        signal_flat = signal.view(-1, length)
        stft_out = torch.stft(signal_flat, n_fft=64, return_complex=True)
        spectrogram = torch.abs(stft_out).view(batch, ch, stft_out.size(1), stft_out.size(2))
        features = self.cnn2d(spectrogram).flatten(1)
        return self.fc(features)

# ---------------------------------------------------------
# 6. Spectral Attention Network
# Applies Self-Attention directly over the frequency bins.
# ---------------------------------------------------------
class SpectralAttentionNet(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(SpectralAttentionNet, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=2, batch_first=True)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, signal):
        mag = torch.abs(torch.fft.rfft(signal, dim=-1))
        mag = mag.permute(0, 2, 1) # (batch, seq, channels)
        attn_out, _ = self.attn(mag, mag, mag)
        features = attn_out.mean(dim=1) # Average over frequency bins
        return self.fc(features)

# ---------------------------------------------------------
# 7. Complex-Valued Spectral CNN (Simulated)
# Treats real and imaginary parts as separate but bound channels.
# ---------------------------------------------------------
class ComplexSpectralCNN(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(ComplexSpectralCNN, self).__init__()
        self.conv_real = nn.Conv1d(channels, 32, 3)
        self.conv_imag = nn.Conv1d(channels, 32, 3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal):
        fft_x = torch.fft.rfft(signal, dim=-1)
        r, i = fft_x.real, fft_x.imag
        # Complex convolution approximation: (X_r*W_r - X_i*W_i) + j(X_r*W_i + X_i*W_r)
        out_r = self.conv_real(r) - self.conv_imag(i)
        out_i = self.conv_real(i) + self.conv_imag(r)
        pooled = torch.cat([out_r.mean(dim=-1), out_i.mean(dim=-1)], dim=1)
        return self.fc(pooled)

# ---------------------------------------------------------
# 8. Dilated Fourier Network (DRFN)
# Uses dilatations on the frequency spectrum to capture wide harmonic bands.
# ---------------------------------------------------------
class DilatedFourierCNN(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(DilatedFourierCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 3, dilation=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, dilation=4), nn.ReLU(), # Captures distant harmonics
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal):
        mag = torch.abs(torch.fft.rfft(signal, dim=-1))
        return self.fc(self.net(mag).flatten(1))

# ---------------------------------------------------------
# 9. Squeeze-and-Excitation Fourier Net (SE-Fourier)
# Recalibrates channel weights based on spectral density.
# ---------------------------------------------------------
class SE_FourierNet(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(SE_FourierNet, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(channels, channels // 2), nn.ReLU(),
            nn.Linear(channels // 2, channels), nn.Sigmoid()
        )
        self.conv = nn.Conv1d(channels, 32, 5, stride=2)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, signal):
        mag = torch.abs(torch.fft.rfft(signal, dim=-1))
        # SE block on average spectral density per channel
        se_weight = self.se(mag.mean(dim=-1)).unsqueeze(2) 
        mag = mag * se_weight
        out = F.relu(self.conv(mag))
        return self.fc(out.mean(dim=-1))

# ---------------------------------------------------------
# 10. Fourier Residual Network (Fourier ResNet)
# Applies residual mappings in the frequency domain to prevent degradation.
# ---------------------------------------------------------
class FourierResidualNet(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(FourierResidualNet, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, signal):
        mag = torch.abs(torch.fft.rfft(signal, dim=-1))
        res = mag
        x = F.relu(self.conv1(mag))
        x = self.conv2(x) + res # Residual connection on spectrum
        return self.fc(x.mean(dim=-1))

# ---------------------------------------------------------
# 11. Wavelet Packet Simulated CNN
# Emulates sub-band decomposition via grouped pooling before convolution.
# ---------------------------------------------------------
class WaveletSimCNN(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(WaveletSimCNN, self).__init__()
        self.low_pass = nn.AvgPool1d(2)
        self.high_pass = nn.MaxPool1d(2) # Proxy for high-frequency details
        self.conv = nn.Conv1d(channels * 2, 64, 3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal):
        low = self.low_pass(signal)
        high = signal[:, :, :low.size(2)] - low # Simulated detail coefficients
        combined = torch.cat([low, high], dim=1)
        features = self.conv(combined).mean(dim=-1)
        return self.fc(features)

# ---------------------------------------------------------
# 12. Gabor Spectral Initialized Net
# Convolutions designed to act as localized time-frequency filters.
# ---------------------------------------------------------
class GaborSpectralNet(nn.Module):
    def __init__(self, channels=4, num_classes=4):
        super(GaborSpectralNet, self).__init__()
        self.conv = nn.Conv1d(channels, 32, kernel_size=15, padding=7)
        # Learnable filters will naturally converge to Gabor-like wavelets
        self.fc = nn.Linear(32, num_classes)

    def forward(self, signal):
        mag = torch.abs(torch.fft.rfft(signal, dim=-1))
        filtered = F.relu(self.conv(mag))
        return self.fc(filtered.mean(dim=-1))
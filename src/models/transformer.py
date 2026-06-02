import torch
import torch.nn as nn

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
import torch
import torch.nn as nn

class FaultClassifier(nn.Module):
    """Proposed Core Network Architecture (WGAN+CNN Classifier)"""
    def __init__(self, num_classes=4, sequence_length=1024):
        super(FaultClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        reduced_len = sequence_length // 16 
        self.fc = nn.Linear(64 * reduced_len, num_classes)

    def forward(self, signal, return_features=False):
        out = self.model(signal)
        features = out.view(out.shape[0], -1)
        logits = self.fc(features) 
        if return_features:
            return logits, features
        return logits

class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=4, sequence_length=1024, embed_dim=50):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.init_size = sequence_length // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim + embed_dim, 128 * self.init_size))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_embedding(labels)
        x = torch.cat([z, c], dim=1)
        out = self.l1(x)
        out = out.view(out.shape[0], 128, self.init_size)
        return self.conv_blocks(out)

class ConditionalCritic(nn.Module):
    def __init__(self, num_classes=4, sequence_length=1024, embed_dim=50):
        super(ConditionalCritic, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.fc_embed = nn.Linear(embed_dim, sequence_length)
        
        self.model = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm([128, sequence_length // 4]), 
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.adv_layer = nn.Linear(128 * (sequence_length // 4), 1)

    def forward(self, signal, labels):
        c = self.label_embedding(labels)
        c = self.fc_embed(c).unsqueeze(1) 
        d_in = torch.cat((signal, c), dim=1) 
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        return self.adv_layer(out)
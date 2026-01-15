import torch
import torch.nn as nn
import torch.nn.functional as F

class IDSModelHybrid(nn.Module):
    def __init__(self, seq_len, feat_dim, num_classes, use_attention=True):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.use_attention = use_attention

        self.conv1 = nn.Conv1d(feat_dim, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        if use_attention:
            self.attn = nn.Linear(256, 1)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            context = torch.mean(lstm_out, dim=1)
            attn_weights = None

        out = self.fc(context)
        return out, out, attn_weights


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

        # CNN feature extractor
        self.conv1 = nn.Conv1d(in_channels=feat_dim, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)

        # LSTM temporal model
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        # Optional attention
        if use_attention:
            self.attn = nn.Linear(128 * 2, 1)

        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq, feat)
        x = x.permute(0, 2, 1)           # (batch, feat, seq)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)           # (batch, seq, 256)

        lstm_out, _ = self.lstm(x)       # (batch, seq, 256)
        if self.use_attention:
            attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            context = torch.mean(lstm_out, dim=1)
            attn_weights = None

        out = self.fc(context)
        return out, out, attn_weights

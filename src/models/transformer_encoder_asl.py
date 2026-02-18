import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [seq_len, batch, d_model]
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TransformerASLEncoder(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 3, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout, max_len=2000)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=False, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """Forward pass.

        x: [B, T, input_dim]
        lengths: [B] lengths of each sequence (number of valid frames)
        Returns logits: [B, num_classes]
        """
        device = x.device
        B, T, _ = x.shape
        x = self.input_proj(x)  # [B, T, d_model]
        # transformer expects [T, B, E]
        x = x.permute(1, 0, 2)

        x = self.pos_enc(x)

        # src_key_padding_mask: True at positions that should be masked (padded)
        arange = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        src_key_padding_mask = arange >= lengths.unsqueeze(1)

        out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        out = out.permute(1, 0, 2)  # [B, T, E]

        # mean-pool over valid positions
        mask = (~src_key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
        summed = (out * mask).sum(dim=1)  # [B, E]
        lengths_clamped = lengths.clamp(min=1).unsqueeze(1).float()
        pooled = summed / lengths_clamped

        logits = self.classifier(pooled)
        return logits

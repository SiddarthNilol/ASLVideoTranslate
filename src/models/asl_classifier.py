import torch
import torch.nn as nn

class GlossClassifier(nn.Module):
    def __init__(self, vjepa_dim = 1408, hidden_dim = 512, num_classes = 300, dropout = 0.1):
        super().__init__()
        self.vjepa_size = vjepa_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.attention_score = nn.Linear(self.vjepa_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(self.vjepa_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )

        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.attention_score.weight)
        if self.attention_score.bias is not None:
            nn.init.zeros_(self.attention_score.bias)

        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, vjepa_features: torch.Tensor):
        # x: [B, T, D]
        attn_scores = self.attention_score(vjepa_features).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T]
        weighted_sum = (vjepa_features * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        logits = self.classifier(weighted_sum)  # [B, num_classes]
        return logits

    def forward_with_attention(self, vjepa_features: torch.Tensor):
        attn_scores = self.attention_score(vjepa_features).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T]
        weighted_sum = torch.sum(vjepa_features * attn_weights.unsqueeze(-1), dim=1)  # [B, D]
        logits = self.classifier(weighted_sum)  # [B, num_classes]
        return logits, attn_weights
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
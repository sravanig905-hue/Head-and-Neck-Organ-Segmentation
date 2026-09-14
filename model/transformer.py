import torch
import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            embed_dim
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(
            embed_dim
        )

        hidden_dim = int(
            embed_dim * mlp_ratio
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                embed_dim,
                hidden_dim
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                embed_dim
            ),

            nn.Dropout(dropout)
        )


    def forward(self, x):

        # Self-attention
        normalized = self.norm1(x)

        attention_output, _ = (
            self.attention(
                normalized,
                normalized,
                normalized
            )
        )

        x = x + attention_output

        # Feed-forward network
        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        depth=2,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.blocks = nn.ModuleList([

            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )

            for _ in range(depth)
        ])


    def forward(self, x):

        for block in self.blocks:

            x = block(x)

        return x
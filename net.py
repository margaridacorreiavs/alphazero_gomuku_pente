import torch
import torch.nn as nn
import torch.nn.functional as F


class GomokuNet(nn.Module):
    """
    Pequena rede tipo AlphaZero:
    - input: (batch, 3, 15, 15)
        canal 0 -> pedras do jogador atual
        canal 1 -> pedras do adversário
        canal 2 -> plano constante (+1 ou -1) para indicar quem é o jogador atual
    - output:
        policy_logits: (batch, 225)  -> probabilidades sobre as 225 casas
        value: (batch, 1)            -> valor em [-1, 1]
    """

    def __init__(self, board_size: int = 15, in_channels: int = 3):
        super().__init__()
        self.board_size = board_size

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Cabeça de política
        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size),
        )

        # Cabeça de valor
        self.value_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # força saída para [-1, 1]
        )

    def forward(self, x):
        # x: (batch, 3, 15, 15)
        h = self.conv_block(x)
        policy_logits = self.policy_head(h)   # (batch, 225)
        value = self.value_head(h)            # (batch, 1)
        return policy_logits, value

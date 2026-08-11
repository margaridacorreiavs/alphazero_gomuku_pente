import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)
        return out


class ResNetGomoku(nn.Module):
    """
    ResNet compacta (10 camadas) para AlphaZero em 15x15.
    Entrada: (batch, 3, 15, 15)
    Saída:
        - policy_logits: (batch, 225)
        - value: (batch, 1)
    """

    def __init__(self, board_size=15, in_channels=3, channels=64, num_blocks=6):
        super().__init__()

        self.board_size = board_size

        # Camada inicial
        self.conv_input = nn.Conv2d(
            in_channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(channels)

        # Blocos residuais
        self.res_layers = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Cabeça da política
        self.policy_conv = nn.Conv2d(
            channels, 2, kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # Cabeça do valor
        self.value_conv = nn.Conv2d(
            channels, 1, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Entrada
        x = F.relu(self.bn_input(self.conv_input(x)))

        # ResNet blocks
        x = self.res_layers(x)

        # ---------- Política ----------
        p = self.policy_conv(x)
        p = F.relu(self.policy_bn(p))
        p = torch.flatten(p, 1)
        policy_logits = self.policy_fc(p)

        # ---------- Valor ----------
        v = self.value_conv(x)
        v = F.relu(self.value_bn(v))
        v = torch.flatten(v, 1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

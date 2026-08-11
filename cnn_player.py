from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from net import GomokuNet
from board_encoding import board_to_tensor, BOARD_SIZE

Coord = Tuple[int, int]


def _opponent(player: int) -> int:
    return 2 if player == 1 else 1


class Player:
    """
    Player controlado por CNN:
    - mesma assinatura do Player mcts
    - funciona para 'gomoku' e 'pente' 
    """

    def __init__(
        self,
        rules: str,
        board_size: int,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.0,
    ):
        rules = rules.lower()
        if board_size != BOARD_SIZE:
            raise ValueError("Este player CNN só suporta tabuleiros 15x15.")
        if rules not in ("gomoku", "pente"):
            raise ValueError("rules must be 'gomoku' or 'pente'")

        self.rules = rules
        self.board_size = board_size
        self.temperature = temperature

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # se não passares model_path, usa 'gomoku_cnn.pt' ou 'pente_cnn.pt'
        if model_path is None:
            model_path = f"{self.rules}_cnn.pt"

        self.net = GomokuNet(board_size=board_size, in_channels=3).to(self.device)
        self.net.eval()

        try:
            state = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(state)
            print(f"[CNNPlayer] Modelo ({self.rules}) carregado de {model_path}")
        except FileNotFoundError:
            print(f"[CNNPlayer] AVISO: ficheiro {model_path} não encontrado, "
                  "o comportamento será quase aleatório.")

    def play(
        self,
        board: List[List[int]],
        turn_number: int,
        last_opponent_move: Optional[Coord],
    ) -> Coord:
        my_id = 1 if turn_number % 2 == 0 else 2

        x = board_to_tensor(board, my_id, self.board_size).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.net(x)
            logits = policy_logits[0]

        board_arr = np.asarray(board, dtype=np.int8)
        empty_mask = (board_arr.reshape(-1) == 0)

        logits_np = logits.cpu().numpy()
        logits_np[~empty_mask] = -1e9
        logits_t = torch.from_numpy(logits_np)

        if self.temperature > 0:
            probs = torch.softmax(logits_t / self.temperature, dim=0).numpy()
        else:
            probs = torch.softmax(logits_t, dim=0).numpy()

        if not np.isfinite(probs).all() or probs.sum() <= 0:
            empties = np.where(empty_mask)[0]
            idx = int(np.random.choice(empties))
        else:
            probs = probs / probs.sum()
            idx = int(np.random.choice(len(probs), p=probs))

        r = idx // self.board_size
        c = idx % self.board_size
        return int(r), int(c)


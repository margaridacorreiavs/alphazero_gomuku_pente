# board_encoding.py
from typing import List
import numpy as np
import torch

BOARD_SIZE = 15

def board_to_tensor(board: List[List[int]], my_id: int, board_size: int = BOARD_SIZE) -> torch.Tensor:
    """
    board: lista de listas com 0 (vazio), 1 (jogador1), 2 (jogador2)
    my_id: 1 ou 2 (quem está a jogar nesta posição)
    retorno: tensor shape (1, 3, board_size, board_size)
        canal 0 -> pedras do jogador atual
        canal 1 -> pedras do adversário
        canal 2 -> plano constante (+1 ou -1) para indicar o lado do jogador atual
    """
    arr = np.asarray(board, dtype=np.int8)
    if arr.shape != (board_size, board_size):
        raise ValueError(f"Esperava tabuleiro {board_size}x{board_size}, recebi {arr.shape}")

    opp_id = 2 if my_id == 1 else 1

    my_plane = (arr == my_id).astype(np.float32)
    opp_plane = (arr == opp_id).astype(np.float32)
    turn_plane = np.full_like(my_plane, 1.0 if my_id == 1 else -1.0, dtype=np.float32)

    x = np.stack([my_plane, opp_plane, turn_plane], axis=0)  # (3, 15, 15)
    x = np.expand_dims(x, axis=0)                            # (1, 3, 15, 15)

    return torch.from_numpy(x)

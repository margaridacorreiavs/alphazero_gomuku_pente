# play_vs_mcts.py
from typing import List, Tuple, Optional

import os
import sys
import numpy as np

# garantir imports locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts_player import (
    Player as MCTSPlayer,
    _evaluate_winner,
    _apply_move_inplace,
    GAME_MODES,
    BOARD_SIZE,
)
from mcts_alphazero import Player as AlphaZeroPlayer
from mcts_alphazero_resnet import Player as AlphaZeroPlayerResnet
from cnn_player import Player as CNNPlayer
from resnet_player import Player as ResNetPlayer

Coord = Tuple[int, int]


def print_board(board: List[List[int]]):
    print("   " + " ".join(f"{c:2d}" for c in range(BOARD_SIZE)))
    for r in range(BOARD_SIZE):
        row_chars = []
        for c in range(BOARD_SIZE):
            v = board[r][c]
            if v == 0:
                ch = "."
            elif v == 1:
                ch = "X"
            else:
                ch = "O"
            row_chars.append(ch)
        print(f"{r:2d} " + " ".join(row_chars))


def main(rules: str = "gomoku"):
    rules = rules.lower()
    if rules not in ("gomoku", "pente"):
        raise ValueError("rules must be 'gomoku' or 'pente'")

    board: List[List[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    last_move: Optional[Coord] = None
    turn = 0

    # vetor de capturas acumuladas (só interessa em Pente)
    captures = np.zeros(2, dtype=np.int16)

    # CNN / AlphaZero / etc.
    p1 = AlphaZeroPlayer(rules=rules, board_size=BOARD_SIZE)
    p2 = AlphaZeroPlayerResnet(rules=rules, board_size=BOARD_SIZE)
    # p2 = MCTSPlayer(rules=rules, board_size=BOARD_SIZE, iterations=400)
    #p1 = CNNPlayer(rules=rules, board_size=BOARD_SIZE)
    #p2 = ResNetPlayer(rules=rules, board_size=BOARD_SIZE)

    print(f"Começa o jogo ({rules}): X = Jogador 1, O = Jogador 2")

    while True:
        current_player = p1 if turn % 2 == 0 else p2
        player_id = 1 if turn % 2 == 0 else 2

        r, c = current_player.play(board, turn, last_opponent_move=last_move)
        if board[r][c] != 0:
            raise RuntimeError(
                f"Jogador {player_id} tentou jogar numa casa ocupada ({r}, {c})"
            )

        if rules == "pente":
            # aplicar capturas corretamente no tabuleiro real,
            # usando o MESMO vetor de capturas ao longo do jogo
            board_arr = np.array(board, dtype=np.int8)

            _apply_move_inplace(
                board_arr,
                captures,
                player_id,
                r,
                c,
                GAME_MODES[rules],
            )

            board = board_arr.tolist()
        else:
            # gomoku: só coloca a peça
            board[r][c] = player_id

        last_move = (r, c)

        print(f"\nTurno {turn} | Jogador {player_id} jogou em ({r}, {c})")
        print_board(board)
        if rules == "pente":
            print(f"Capturas: X={captures[0]}  O={captures[1]}")

        arr = np.asarray(board, dtype=np.int8)

        # em Pente, a condição de vitória por capturas usa ESTE vetor
        winner = _evaluate_winner(arr, captures, GAME_MODES[rules])

        if winner != -1:
            if winner == 0:
                print("\nEmpate!")
            else:
                print(f"\nVitória do jogador {winner}")
            break

        turn += 1
        if turn >= BOARD_SIZE * BOARD_SIZE:
            print("\nTabuleiro cheio, empate.")
            break


if __name__ == "__main__":
    # MUDAR AQUI PARA TESTAR GOMOKU OU PENTE
    main(rules="pente")  # ou main(rules="gomoku")


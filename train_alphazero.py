# train_alphazero.py
from typing import List, Tuple, Optional

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from tqdm import tqdm


# mudar device consoante computador
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("A usar MPS")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("A usar GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("A usar CPU")

# garantir que encontra os módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# AlphaZero MCTS (usa a rede internamente para avaliar)
from mcts_alphazero import (
    Player as AZPlayer,
    _evaluate_winner,
    _apply_move_inplace,
    GAME_MODES,
    BOARD_SIZE,
)

from board_encoding import board_to_tensor
from net import GomokuNet          # CNN “clássica”
from resnet import ResNetGomoku   # ResNet compacta

Coord = Tuple[int, int]


# -----------------------------------------------------------
#   Self-play AlphaZero vs AlphaZero
# -----------------------------------------------------------
def play_one_game(
    num_iterations: int = 1000,
    rules: str = "gomoku",
) -> Tuple[list, int]:
    """
    Joga um jogo AZPlayer vs AZPlayer.
    Devolve:
      - history: lista de (board_antes, player_id, move)
      - winner: 0 (empate), 1 ou 2
    """

    rules = rules.lower()
    if rules not in ("gomoku", "pente"):
        raise ValueError("rules must be 'gomoku' or 'pente'")

    # Cada jogador AlphaZero carrega a rede do disco (rules_cnn.pt)
    # ou usa a random init caso ainda não exista.
    p1 = AZPlayer(rules=rules, board_size=BOARD_SIZE, iterations=num_iterations)
    p2 = AZPlayer(rules=rules, board_size=BOARD_SIZE, iterations=num_iterations)

    board: List[List[int]] = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    last_move: Optional[Coord] = None
    turn = 0
    history = []

    # Para Pente: acompanhar capturas reais, para o _evaluate_winner
    captures_arr = np.zeros(2, dtype=np.int16)

    while True:
        current_player = p1 if turn % 2 == 0 else p2
        player_id = 1 if turn % 2 == 0 else 2

        move = current_player.play(board, turn, last_move)
        r, c = move

        if board[r][c] != 0:
            # Algo correu muito mal: jogada inválida
            # Terminamos o jogo e marcamos empate.
            return history, 0

        # Guardar estado *antes* da jogada
        history.append(([row[:] for row in board], player_id, move))

        if rules == "pente":
            # Aplicar capturas reais no tabuleiro para refletir as regras de Pente
            board_arr = np.array(board, dtype=np.int8)
            _apply_move_inplace(
                board_arr,
                captures_arr,
                player_id,
                r,
                c,
                GAME_MODES[rules],
            )
            board = board_arr.tolist()
        else:
            # Gomoku: só coloca a peça
            board[r][c] = player_id

        last_move = move
        turn += 1

        arr = np.asarray(board, dtype=np.int8)
        winner = _evaluate_winner(arr, captures_arr, GAME_MODES[rules])

        # Fim do jogo
        if winner != -1 or turn >= BOARD_SIZE * BOARD_SIZE:
            if winner == -1:
                winner = 0
            return history, int(winner)


# -----------------------------------------------------------
#   Dataset (X, Pi, Z)
# -----------------------------------------------------------
def generate_dataset(
    num_games: int = 10,
    rules: str = "gomoku",
    mcts_iterations: int = 1000,
):
    """
    Gera dados a partir de jogos AlphaZero vs AlphaZero.

    Retorna:
      X  : (N, 3, 15, 15)  tensores de estado
      Pi : (N, 225)        one-hot da jogada escolhida pelo MCTS
      Z : (N, 1)          resultado do jogo na perspetiva do jogador da jogada
    """
    states = []
    policy_targets = []
    value_targets = []

    for g in range(num_games):
        print(f"[self-play-{rules}] Jogo {g+1}/{num_games}")
        history, winner = play_one_game(
            num_iterations=mcts_iterations,
            rules=rules,
        )

        for board_before, player_id, move in history:
            # Resultado z: 1 se jogador da jogada ganhou, -1 se perdeu, 0 se empate
            if winner == 0:
                z = 0.0
            elif winner == player_id:
                z = 1.0
            else:
                z = -1.0

            # Estado codificado (tabuleiro + canal do jogador)
            x = board_to_tensor(board_before, player_id, BOARD_SIZE)[0]
            states.append(x)

            # Política alvo: one-hot da jogada efetuada (podemos
            # futuramente substituir por visitas do MCTS)
            idx = move[0] * BOARD_SIZE + move[1]
            pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            pi[idx] = 1.0

            policy_targets.append(pi)
            value_targets.append([z])

    X = torch.stack(states)
    Pi = torch.tensor(np.array(policy_targets))
    Z = torch.tensor(np.array(value_targets), dtype=torch.float32)
    return X, Pi, Z


# -----------------------------------------------------------
#   Treinar um modelo (CNN ou ResNet)
# -----------------------------------------------------------
def train_single_model(
    net: torch.nn.Module,
    X: torch.Tensor,
    Pi: torch.Tensor,
    Z: torch.Tensor,
    save_path: str,
    device: torch.device,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    desc: str = "",
):
    dataset = TensorDataset(X, Pi, Z)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for xb, pib, zb in tqdm(loader, desc=f"{desc} | Epoch {epoch+1}/{epochs}"):
            xb = xb.to(device)
            pib = pib.to(device)
            zb = zb.to(device)

            policy_logits, value_pred = net(xb)

            # Loss de valor (MSE)
            value_loss = (value_pred - zb).pow(2).mean()

            # Loss de política (cross-entropy com alvo Pi)
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(pib * log_probs).sum(dim=1).mean()

            loss = value_loss + policy_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * xb.size(0)

        avg_loss = running_loss / len(dataset)
        print(f"[{desc}] Epoch {epoch+1}: loss médio = {avg_loss:.4f}")

    torch.save(net.state_dict(), save_path)
    print(f"[{desc}] Modelo guardado em {save_path}")


# -----------------------------------------------------------
#   Loop principal: gera dados e treina CNN + ResNet
# -----------------------------------------------------------
def train(
    num_games: int = 20,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    rules: str = "gomoku",
    mcts_iterations: int = 1000,
):
    """
    Gera um dataset de self-play AlphaZero vs AlphaZero
    (usando o modelo atual em disco como engine)
    e treina / continua a treinar:

      - {rules}_resnet.pt  (ResNetGomoku)
      - {rules}_cnn.pt     (GomokuNet)
    """

    rules = rules.lower()
    if rules not in ("gomoku", "pente"):
        raise ValueError("rules must be 'gomoku' or 'pente'")

    # 1) Gerar dataset com a política atual (AZPlayer carrega modelo {rules}_cnn.pt)
    X, Pi, Z = generate_dataset(
        num_games=num_games,
        rules=rules,
        mcts_iterations=mcts_iterations,
    )

    # 2) ResNet
    resnet_path = f"{rules}_resnet.pt"
    resnet = ResNetGomoku(board_size=BOARD_SIZE, in_channels=3).to(device)
    if os.path.exists(resnet_path):
        res_state = torch.load(resnet_path, map_location=device)
        resnet.load_state_dict(res_state)
        print(f"[{rules}_resnet] Checkpoint carregado de {resnet_path}")
    else:
        print(f"[{rules}_resnet] Nenhum checkpoint encontrado, treino de raiz.")

    train_single_model(
        resnet,
        X,
        Pi,
        Z,
        resnet_path,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        desc=f"{rules}_resnet",
    )

    # 3) CNN
    cnn_path = f"{rules}_cnn.pt"
    cnn = GomokuNet(board_size=BOARD_SIZE, in_channels=3).to(device)
    if os.path.exists(cnn_path):
        cnn_state = torch.load(cnn_path, map_location=device)
        cnn.load_state_dict(cnn_state)
        print(f"[{rules}_cnn] Checkpoint carregado de {cnn_path}")
    else:
        print(f"[{rules}_cnn] Nenhum checkpoint encontrado, treino de raiz.")

    train_single_model(
        cnn,
        X,
        Pi,
        Z,
        cnn_path,
        device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        desc=f"{rules}_cnn",
    )


# -----------------------------------------------------------
#   LOOP PRINCIPAL DE CICLOS  ← ÚNICA ADIÇÃO
# -----------------------------------------------------------
def train_loop(
    rules: str = "gomoku",
    num_cycles: int = 12,        # nº de ciclos
    games_per_cycle: int = 30,   # jogos de self-play por ciclo
    epochs: int = 3,             # epochs de treino por ciclo
    batch_size: int = 64,
    lr: float = 1e-3,
    mcts_iterations: int = 1000,
):
    """
    Executa vários ciclos de:
      1) self-play (games_per_cycle jogos)
      2) treino CNN + ResNet (epochs epochs)
      3) guardar checkpoints
    """
    for cycle in range(num_cycles):
        print("\n===================================")
        print(f"   CICLO {cycle+1}/{num_cycles}")
        print("===================================\n")

        train(
            num_games=games_per_cycle,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            rules=rules,
            mcts_iterations=mcts_iterations,
        )


# -----------------------------------------------------------
#   MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    # Mudar aqui entre "gomoku" e "pente"
    RULES = "gomoku"

    # Agora usamos o loop de ciclos em vez de uma única chamada a train()
    train_loop(
        rules=RULES,
        num_cycles=10000,          # nº de ciclos totais
        games_per_cycle=40,     # jogos por ciclo
        epochs=4,               # epochs por ciclo
        batch_size=64,
        lr=1e-3,
        mcts_iterations=800,
    )

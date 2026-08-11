# mcts_alphazero.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# garantir que encontra os módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mcts_player import (
    BOARD_SIZE,
    GAME_MODES,
    _evaluate_winner,
    _apply_move_inplace,
    _opponent,
    _five_in_row,
)
from board_encoding import board_to_tensor
from net import GomokuNet

Coord = Tuple[int, int]


# ----------------------------------------------------------------------
#  Heurísticas AlphaZero (apenas antes do MCTS)
# ----------------------------------------------------------------------
def _creates_double_threat(board: np.ndarray, player: int, r: int, c: int) -> bool:
    """
    Verifica se, ao jogar em (r,c), o player fica com >=2 casas de vitória
    (double-threat) em jogadas seguintes. Ignora capturas, só 5-em-linha.
    """
    if board[r, c] != 0:
        return False

    board[r, c] = player
    threats = 0

    empties2 = np.argwhere(board == 0)
    for r2, c2 in empties2:
        board[r2, c2] = player
        if _five_in_row(board, player):
            threats += 1
        board[r2, c2] = 0
        if threats >= 2:
            board[r, c] = 0
            return True

    board[r, c] = 0
    return False


def _forced_move_az(
    board: np.ndarray,
    captures: np.ndarray,
    player: int,
    game_mode: int,
) -> Optional[Coord]:
    """
    Heurísticas usadas APENAS antes do MCTS:

    1) Se podemos ganhar já (5 em linha ou, em Pente, vitória por capturas) → jogar.
    2) Se o adversário pode ganhar já → bloquear.
    3) Se temos jogada que cria double-threat ofensiva → jogar.
    4) Se o adversário tem jogada que criaria double-threat → bloquear.
    5) (Pente) Se nenhuma acima se aplica e podemos capturar 2 peças → capturar.
    6) Caso contrário → None (MCTS decide).
    """
    opp = _opponent(player)
    empties = np.argwhere(board == 0)

    # 1) Vitória imediata nossa (inclui vitória por captura em Pente)
    for r, c in empties:
        r_i, c_i = int(r), int(c)
        b = board.copy()
        caps = captures.copy()
        _apply_move_inplace(b, caps, player, r_i, c_i, game_mode)
        if _evaluate_winner(b, caps, game_mode) == player:
            return (r_i, c_i)

    # 2) Bloquear vitória imediata do adversário
    for r, c in empties:
        r_i, c_i = int(r), int(c)
        b = board.copy()
        caps = captures.copy()
        _apply_move_inplace(b, caps, opp, r_i, c_i, game_mode)
        if _evaluate_winner(b, caps, game_mode) == opp:
            # Jogamos nós em (r_i, c_i) para impedir
            return (r_i, c_i)

    # 3) Double-threat ofensiva
    for r, c in empties:
        r_i, c_i = int(r), int(c)
        if _creates_double_threat(board, player, r_i, c_i):
            return (r_i, c_i)

    # 4) Bloquear double-threat do adversário
    for r, c in empties:
        r_i, c_i = int(r), int(c)
        if _creates_double_threat(board, opp, r_i, c_i):
            # Se o adversário jogar aqui, cria double-threat;
            # por isso jogamos nós aqui para bloquear.
            return (r_i, c_i)

    # 5) Heurística específica de Pente: capturar se possível
    if game_mode == GAME_MODES["pente"]:
        for r, c in empties:
            r_i, c_i = int(r), int(c)
            b = board.copy()
            caps = captures.copy()
            before = int(caps[player - 1])
            _apply_move_inplace(b, caps, player, r_i, c_i, game_mode)
            after = int(caps[player - 1])
            if after > before:
                # Esta jogada captura pelo menos 2 peças
                return (r_i, c_i)

    return None


# ----------------------------------------------------------------------
#  Nó de MCTS estilo AlphaZero
# ----------------------------------------------------------------------
class AZNode:
    __slots__ = (
        "board",
        "captures",
        "to_move",
        "parent",
        "move",
        "children",
        "visits",
        "value_sum",
        "prior",
    )

    def __init__(
        self,
        board: np.ndarray,
        captures: np.ndarray,
        to_move: int,
        parent: Optional["AZNode"] = None,
        move: Optional[Coord] = None,
        prior: float = 0.0,
    ) -> None:
        self.board = board
        self.captures = captures
        self.to_move = to_move
        self.parent = parent
        self.move = move
        self.children: Dict[Coord, "AZNode"] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = float(prior)

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class Player:
    """
    AlphaZero-style player:
    - Aceita rede externa (necessário para treino AlphaZero)
    - search_with_policy() devolve (move, pi_vector)
    - Aplica heurísticas obrigatórias ANTES do MCTS (não mexe na árvore)
    """

    def __init__(
        self,
        rules: str,
        board_size: int,
        net: Optional[torch.nn.Module] = None,   # pode receber rede externa
        model_path: Optional[str] = None,
        iterations: int = 800,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        use_dirichlet_noise: bool = False,
        dirichlet_alpha: float = 0.03,
        dirichlet_eps: float = 0.25,
    ) -> None:

        rules = rules.lower()
        if board_size != BOARD_SIZE:
            raise ValueError("This AlphaZero-style player only supports 15x15 boards.")
        if rules not in GAME_MODES:
            raise ValueError("rules must be 'gomoku' or 'pente'")

        self.rules = rules
        self.board_size = board_size
        self.game_mode_flag = GAME_MODES[rules]

        self.iterations = iterations
        self.c_puct = c_puct
        self.temperature = temperature
        self._seed = seed

        self.use_dirichlet_noise = use_dirichlet_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._rng = np.random.RandomState(self._seed)

        # Se passar uma rede externa (do treino AlphaZero), usa essa.
        if net is not None:
            self.net = net.to(self.device)
            self.net.eval()
            self._external_net = True
        else:
            # Caso contrário, carrega {rules}_cnn.pt como antes
            if model_path is None:
                model_path = f"{rules}_cnn.pt"

            self.net = GomokuNet(board_size=BOARD_SIZE, in_channels=3).to(self.device)

            if os.path.exists(model_path):
                self.net.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[alphazero] Modelo carregado de {model_path}")
            else:
                print(f"[alphazero] Aviso: modelo {model_path} não encontrado.")
            self.net.eval()
            self._external_net = False

    # ------------------------------------------------------------------
    #  AlphaZero: devolve jogada + vetor de política (visitas normalizadas)
    # ------------------------------------------------------------------
    def search_with_policy(
        self,
        board: List[List[int]],
        turn_number: int,
        last_opponent_move: Optional[Coord],
    ) -> Tuple[Coord, np.ndarray]:
        my_id = 1 if turn_number % 2 == 0 else 2
        board_arr, captures = self._state_from_view(board, turn_number, my_id)

        # Heurísticas obrigatórias ANTES do MCTS
        forced = _forced_move_az(board_arr.copy(), captures.copy(), my_id, self.game_mode_flag)
        if forced is not None:
            pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            pi[forced[0] * BOARD_SIZE + forced[1]] = 1.0
            return forced, pi

        # Construir root
        root = AZNode(board_arr.copy(), captures.copy(), my_id, None, None, 1.0)

        # MCTS AlphaZero
        for _ in range(self.iterations):
            self._run_simulation(root)

        # Vetor pi com base nas visitas
        pi = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        for move, child in root.children.items():
            idx = move[0] * BOARD_SIZE + move[1]
            pi[idx] = child.visits

        if pi.sum() > 0:
            pi /= pi.sum()
        else:
            pi += 1.0 / (BOARD_SIZE * BOARD_SIZE)

        move = self._select_move_from_root(root)
        return move, pi

    # Mantemos play() para compatibilidade (só devolve a jogada)
    def play(
        self,
        board: List[List[int]],
        turn_number: int,
        last_opponent_move: Optional[Coord],
    ) -> Coord:
        move, _ = self.search_with_policy(board, turn_number, last_opponent_move)
        return move

    # ------------------------------------------------------------------
    #  MCTS AlphaZero
    # ------------------------------------------------------------------
    def _run_simulation(self, root: AZNode) -> None:
        node = root

        # Selection
        while node.is_expanded():
            winner = _evaluate_winner(node.board, node.captures, self.game_mode_flag)
            if winner != -1:
                break
            node = self._select_child_puct(node)

        # Avaliação
        winner = _evaluate_winner(node.board, node.captures, self.game_mode_flag)
        if winner != -1:
            if winner == 0:
                value = 0.0
            elif winner == node.to_move:
                value = 1.0
            else:
                value = -1.0
        else:
            value = self._evaluate_and_expand(node)

        # Backprop
        self._backpropagate(node, value)

    def _select_child_puct(self, node: AZNode) -> AZNode:
        total_visits = sum(child.visits for child in node.children.values())
        sqrt_total = math.sqrt(total_visits) if total_visits > 0 else 1.0

        best_score = -1e9
        best_child: Optional[AZNode] = None

        for move, child in node.children.items():
            q = child.q_value
            u = self.c_puct * child.prior * (sqrt_total / (1 + child.visits))
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child  # type: ignore

    def _evaluate_and_expand(self, node: AZNode) -> float:
        board_view = node.board.tolist()
        player_id = int(node.to_move)

        with torch.no_grad():
            state_tensor = board_to_tensor(board_view, player_id, BOARD_SIZE)[0].unsqueeze(0).to(self.device)
            policy_logits, value_pred = self.net(state_tensor)
            value = float(value_pred.item())
            policy_probs = F.softmax(policy_logits[0], dim=0).cpu().numpy()

        empties = np.argwhere(node.board == 0)
        if empties.size == 0:
            return 0.0

        priors: Dict[Coord, float] = {}
        for r, c in empties:
            idx = int(r) * BOARD_SIZE + int(c)
            priors[(int(r), int(c))] = float(policy_probs[idx])

        # Ruído de Dirichlet na root (exploração extra)
        if node.parent is None and self.use_dirichlet_noise:
            moves = list(priors.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            for i, m in enumerate(moves):
                priors[m] = (1 - self.dirichlet_eps) * priors[m] + self.dirichlet_eps * float(noise[i])

        # Normalizar priors
        s = sum(max(p, 0.0) for p in priors.values())
        if s > 0:
            for m in priors:
                priors[m] = max(priors[m], 0.0) / s
        else:
            uniform_p = 1.0 / len(priors)
            for m in priors:
                priors[m] = uniform_p

        # Criar filhos
        for (r, c), p in priors.items():
            new_board = node.board.copy()
            new_captures = node.captures.copy()

            _apply_move_inplace(
                new_board,
                new_captures,
                node.to_move,
                int(r),
                int(c),
                self.game_mode_flag,
            )

            child = AZNode(
                board=new_board,
                captures=new_captures,
                to_move=_opponent(node.to_move),
                parent=node,
                move=(int(r), int(c)),
                prior=p,
            )
            node.children[(int(r), int(c))] = child

        return value

    def _backpropagate(self, node: AZNode, leaf_value: float) -> None:
        current = node
        value = leaf_value
        while current is not None:
            current.visits += 1
            current.value_sum += value
            value = -value
            current = current.parent  # type: ignore

    def _select_move_from_root(self, root: AZNode) -> Coord:
        if not root.children:
            empties = np.argwhere(root.board == 0)
            if empties.size == 0:
                raise RuntimeError("No legal moves.")
            r, c = empties[self._rng.randint(len(empties))]
            return int(r), int(c)

        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float32)

        if self.temperature <= 1e-6:
            idx = int(np.argmax(visits))
            return moves[idx]

        v = visits ** (1.0 / self.temperature)
        v_sum = v.sum()
        if v_sum <= 0:
            idx = int(np.argmax(visits))
            return moves[idx]

        probs = v / v_sum
        idx = int(self._rng.choice(len(moves), p=probs))
        return moves[idx]

    # ----------------------------------------------------------------------
    # Estado (igual à tua versão anterior)
    # ----------------------------------------------------------------------
    def _state_from_view(self, board_view, turn_number, my_id):
        arr = np.asarray(board_view, dtype=np.int8)
        actual = np.ascontiguousarray(arr, dtype=np.int8)

        captures = np.zeros(2, dtype=np.int16)
        if my_id == 1:
            my_moves = (turn_number + 1) // 2
        else:
            my_moves = turn_number // 2
        opp_moves = turn_number - my_moves

        my_stones = int(np.count_nonzero(actual == 1))
        opp_stones = int(np.count_nonzero(actual == 2))

        my_captured = max(0, opp_moves - opp_stones)
        opp_captured = max(0, my_moves - my_stones)

        captures[0] = my_captured
        captures[1] = opp_captured
        return actual, captures

    def _legal_moves_from_array(self, arr):
        empties = np.argwhere(arr == 0)
        return [tuple(map(int, pos)) for pos in empties]

from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np
from numba import njit

Coord = Tuple[int, int]

BOARD_SIZE = 15
CAPTURE_TARGET = 10
DIRS_8 = np.array(
    [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ],
    dtype=np.int8,
)
GAME_MODES = {"gomoku": 0, "pente": 1}


def _opponent(player: int) -> int:
    return 2 if player == 1 else 1


# ================================================================
#  NUMBA ZONE — funções de baixo nível
# ================================================================
@njit
def _board_full(board: np.ndarray) -> bool:
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                return False
    return True


@njit
def _five_in_row(board: np.ndarray, player: int) -> bool:
    target = 5
    N = BOARD_SIZE

    # Horizontal
    for r in range(N):
        run = 0
        for c in range(N):
            if board[r, c] == player:
                run += 1
                if run >= target:
                    return True
            else:
                run = 0

    # Vertical
    for c in range(N):
        run = 0
        for r in range(N):
            if board[r, c] == player:
                run += 1
                if run >= target:
                    return True
            else:
                run = 0

    # Diagonal TL-BR
    for start in range(N):
        run = 0
        r, c = start, 0
        while r < N and c < N:
            if board[r, c] == player:
                run += 1
                if run >= target:
                    return True
            else:
                run = 0
            r += 1
            c += 1

        run = 0
        r, c = 0, start
        while r < N and c < N:
            if board[r, c] == player:
                run += 1
                if run >= target:
                    return True
            else:
                run = 0
            r += 1
            c += 1

    # Diagonal TR-BL
    for start in range(N):
        run = 0
        r, c = start, N - 1
        while r < N and c >= 0:
            if board[r, c] == player:
                run += 1
                if run >= target:
                    return True
            else:
                run = 0
            r += 1
            c -= 1

        run = 0
        r, c = 0, start
        while r < N and c >= 0:
            if board[r, c] == player:
                run += 1
                if run >= target:
                    return True
            else:
                run = 0
            r += 1
            c -= 1

    return False


@njit
def _evaluate_winner(board: np.ndarray, captures: np.ndarray, game_mode: int) -> int:
    if _five_in_row(board, 1):
        return 1
    if _five_in_row(board, 2):
        return 2
    if game_mode == 1:
        if captures[0] >= CAPTURE_TARGET:
            return 1
        if captures[1] >= CAPTURE_TARGET:
            return 2
    if _board_full(board):
        return 0
    return -1


@njit
def _apply_move_inplace(
    board: np.ndarray, captures: np.ndarray, player: int, r: int, c: int, game_mode: int
) -> None:
    board[r, c] = player
    if game_mode == 1:
        opp = 2 if player == 1 else 1
        removed = 0
        for k in range(DIRS_8.shape[0]):
            dr = DIRS_8[k, 0]
            dc = DIRS_8[k, 1]
            r1 = r + dr
            c1 = c + dc
            r2 = r1 + dr
            c2 = c1 + dc
            r3 = r2 + dr
            c3 = c2 + dc
            if (
                0 <= r1 < BOARD_SIZE
                and 0 <= c1 < BOARD_SIZE
                and 0 <= r2 < BOARD_SIZE
                and 0 <= c2 < BOARD_SIZE
                and 0 <= r3 < BOARD_SIZE
                and 0 <= c3 < BOARD_SIZE
            ):
                if (
                    board[r1, c1] == opp
                    and board[r2, c2] == opp
                    and board[r3, c3] == player
                ):
                    board[r1, c1] = 0
                    board[r2, c2] = 0
                    removed += 2
        if removed:
            captures[player - 1] += removed


# ================================================================
#  HEURÍSTICAS DE ALTO NÍVEL (NÃO NUMBA)
# ================================================================
def _creates_double_threat(board: np.ndarray, player: int, r: int, c: int) -> bool:
    """
    Ao jogar em (r,c), jogador cria >=2 casas futuras de vitória?
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


def _forced_move_mcts(
    board: np.ndarray,
    captures: np.ndarray,
    player: int,
    game_mode: int,
) -> Optional[Coord]:
    """
    As 6 heurísticas na ordem pedida:
    1) Vitória imediata.
    2) Bloquear vitória imediata do adversário.
    3) Double-threat ofensiva.
    4) Bloquear double-threat.
    5) (Pente) Captura imediata.
    6) Caso contrário → None.
    """
    opp = _opponent(player)
    empties = np.argwhere(board == 0)

    # 1 — vitória imediata
    for r, c in empties:
        r, c = int(r), int(c)
        b = board.copy()
        caps = captures.copy()
        _apply_move_inplace(b, caps, player, r, c, game_mode)
        if _evaluate_winner(b, caps, game_mode) == player:
            return (r, c)

    # 2 — bloquear vitória imediata do adversário
    for r, c in empties:
        r, c = int(r), int(c)
        b = board.copy()
        caps = captures.copy()
        _apply_move_inplace(b, caps, opp, r, c, game_mode)
        if _evaluate_winner(b, caps, game_mode) == opp:
            return (r, c)

    # 3 — double-threat ofensiva
    for r, c in empties:
        r, c = int(r), int(c)
        if _creates_double_threat(board, player, r, c):
            return (r, c)

    # 4 — bloquear double-threat adversário
    for r, c in empties:
        r, c = int(r), int(c)
        if _creates_double_threat(board, opp, r, c):
            return (r, c)

    # 5 — captura imediata (só Pente)
    if game_mode == GAME_MODES["pente"]:
        for r, c in empties:
            r, c = int(r), int(c)
            b = board.copy()
            caps = captures.copy()
            before = caps[player - 1]
            _apply_move_inplace(b, caps, player, r, c, game_mode)
            if caps[player - 1] > before:
                return (r, c)

    return None


# ================================================================
#  MCTS ORIGINAL
# ================================================================
@njit
def _random_empty_cell(board: np.ndarray) -> Tuple[int, int]:
    empties = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                empties += 1
    if empties == 0:
        return (-1, -1)
    pick = np.random.randint(0, empties)
    seen = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == 0:
                if seen == pick:
                    return (r, c)
                seen += 1
    return (-1, -1)


@njit
def _simulate_rollout(board, captures, player_to_move, game_mode, rollout_limit) -> int:
    b = board.copy()
    caps = captures.copy()
    current = player_to_move
    for _ in range(rollout_limit):
        result = _evaluate_winner(b, caps, game_mode)
        if result != -1:
            return result
        r, c = _random_empty_cell(b)
        if r == -1:
            return 0
        _apply_move_inplace(b, caps, current, r, c, game_mode)
        current = _opponent(current)

    result = _evaluate_winner(b, caps, game_mode)
    if result != -1:
        return result
    return 0


class MCTSNode:
    __slots__ = (
        "board", "captures", "player", "parent", "move",
        "children", "visits", "wins", "untried_moves"
    )

    def __init__(self, board, captures, player_just_moved, rng, parent=None, move=None):
        self.board = board
        self.captures = captures
        self.player = player_just_moved
        self.parent = parent
        self.move = move
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.wins = 0.0

        empties = np.argwhere(board == 0)
        moves = [tuple(map(int, pos)) for pos in empties]
        rng.shuffle(moves)
        self.untried_moves = moves

    def ucb1(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class Player:
    def __init__(
        self,
        rules,
        board_size,
        iterations=None,
        rollout_limit=None,
        exploration=1.41,
        seed=None
    ):
        rules = rules.lower()
        if board_size != BOARD_SIZE:
            raise ValueError("Only 15x15 supported.")
        if rules not in GAME_MODES:
            raise ValueError("rules must be 'gomoku' or 'pente'")

        self.rules = rules
        self.board_size = board_size
        self.iterations = iterations or 2500
        self.rollout_limit = rollout_limit or board_size * board_size
        self.exploration = exploration
        self.game_mode_flag = GAME_MODES[rules]
        self._seed = seed

    def play(self, board, turn_number, last_opponent_move):
        rng_seed = None if self._seed is None else self._seed + turn_number
        rng = random.Random(rng_seed)

        my_id = 1 if turn_number % 2 == 0 else 2
        board_arr, captures = self._state_from_view(board, turn_number, my_id)

        # ============================================================
        # HEURÍSTICAS OBRIGATÓRIAS ANTES DO MCTS
        # ============================================================
        forced = _forced_move_mcts(board_arr.copy(), captures.copy(), my_id, self.game_mode_flag)
        if forced is not None:
            return forced

        # Estado terminal
        result = _evaluate_winner(board_arr, captures, self.game_mode_flag)
        if result != -1:
            legal = self._legal_moves_from_array(board_arr)
            return rng.choice(legal)

        root = MCTSNode(
            board_arr.copy(),
            captures.copy(),
            _opponent(my_id),
            rng,
            parent=None,
            move=None,
        )

        # MCTS
        for _ in range(self.iterations):
            node = root

            # seleção
            while not node.untried_moves and node.children:
                node = max(node.children, key=lambda child: child.ucb1(node.visits, self.exploration))

            # expansão
            if node.untried_moves:
                node = self._expand(node, rng)

            # simulação
            winner = self._rollout(node, _opponent(node.player))

            # backprop
            self._backpropagate(node, winner)

        chosen = self._best_child(root)
        if chosen is None or chosen.move is None:
            legal = self._legal_moves_from_array(board_arr)
            return rng.choice(legal)
        return chosen.move

    # -------------------------------------------------------------
    #  MCTS helpers
    # -------------------------------------------------------------
    def _expand(self, node, rng):
        move = node.untried_moves.pop()
        next_player = _opponent(node.player)
        new_board = node.board.copy()
        new_captures = node.captures.copy()

        _apply_move_inplace(new_board, new_captures, next_player, move[0], move[1], self.game_mode_flag)

        child = MCTSNode(new_board, new_captures, next_player, rng, parent=node, move=move)
        node.children.append(child)
        return child

    def _rollout(self, node, player_to_move):
        return int(
            _simulate_rollout(
                node.board,
                node.captures,
                player_to_move,
                self.game_mode_flag,
                self.rollout_limit,
            )
        )

    def _backpropagate(self, node, winner):
        current = node
        while current is not None:
            current.visits += 1
            if winner == 0:
                current.wins += 0.5
            elif winner == current.player:
                current.wins += 1.0
            current = current.parent

    def _best_child(self, node):
        if not node.children:
            return None
        visited = [c for c in node.children if c.visits > 0]
        target = visited or node.children
        return max(target, key=lambda child: child.wins / child.visits if child.visits else 0.0)

    # -------------------------------------------------------------
    # state_from_view
    # -------------------------------------------------------------
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

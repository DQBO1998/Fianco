
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from fianco import number, Mat, YX, is_capt, win, vdir, Engine
from numpy.typing import NDArray
from numpy import inf, nan
from numba.experimental import jitclass # type: ignore
from numpy.random import default_rng
from tables import *

import numpy as np
import numba as nb # type: ignore
import time


age = 0
def_size = -1
r_ply, r_mat, table = None, None, None


def reset(size: int):
    global age
    global table
    global r_ply
    global r_mat
    global def_size
    age = 0
    if def_size != size:
        def_size = size
        r_ply, r_mat, table = make_tt(def_size, default_rng(42))
    assert table is not None
    table[HEIGHT][:] = -1


def dbg(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrap(*args: Any, **kwargs: Any) -> Any:
        org = set(func.signatures) # type: ignore
        ret = func(*args, **kwargs)
        new = set(func.signatures) # type: ignore
        if new != org:
            print(f'compiled `{func.__name__}` for `{new}`')
        return ret
    return wrap


@jitclass([
    ('end_state', nb.bool[:, :, :]),
    ('board_state', nb.bool[:, :, :]),
    ('move_from', nb.int8[:]),
    ('move_to', nb.int8[:]),
    ('tt_lock', nb.uint32[:]),
    ('tt_move_from', nb.int8[:, :]),
    ('tt_move_to', nb.int8[:, :]),
    ('tt_score', nb.float64[:]),
    ('tt_flag', nb.uint8[:]),
    ('tt_height', nb.int32[:]),
    ('tt_age', nb.uint32[:]),
    ('tt_player', nb.uint8[:]),  # New field for player information
    ('r_mat', nb.uint32[:, :, :]),
    ('r_ply', nb.uint32[:]),
    ('age', nb.uint32),
    ('nodes', nb.uint64),
    ('hits', nb.uint64),
    ('writes', nb.uint64)
]) # type: ignore
class SearchState:
    end_state: Mat
    board_state: Mat
    
    move_from: NDArray[number]
    move_to: NDArray[number]

    tt_lock: TableLock
    tt_move_from: TableMoveFrom
    tt_move_to: TableMoveTo
    tt_score: TableScore
    tt_flag: TableFlag
    tt_height: TableHeight
    tt_age: TableAge
    tt_player: TablePlayer  # New field for player information

    r_mat: RNums
    r_ply: NDArray[Hash]

    age: int

    nodes: int
    hits: int
    writes: int

    def __init__(self, 
                 move_from: YX, move_to: YX,
                 end_state: Mat, board_state: Mat,
                 r_ply: NDArray[Hash], r_mat: RNums, _table: Table,
                 age: int) -> None:
        self.end_state = end_state
        self.board_state = board_state

        self.move_from = move_from
        self.move_to = move_to

        self.tt_lock = _table[LOCK]
        self.tt_move_from = _table[FROM]
        self.tt_move_to = _table[TO]
        self.tt_score = _table[SCORE]
        self.tt_flag = _table[FLAG]
        self.tt_height = _table[HEIGHT]
        self.tt_age = _table[AGE]
        self.tt_player = _table[PLAYER]  # Initialize the new player field

        self.r_mat = r_mat
        self.r_ply = r_ply

        self.age = age

        self.nodes = 0
        self.hits = 0
        self.writes = 0

    def table(self) -> Table:
        return (self.tt_lock, self.tt_move_from, self.tt_move_to,
                self.tt_score, self.tt_flag, self.tt_height, self.tt_age,
                self.tt_player)  # Include tt_player in the returned tuple


@nb.njit # type: ignore
def capts(wrt: int, brd: Mat, frto: NDArray[number]) -> int:
    at = 0
    dy = vdir(wrt)
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[wrt, y, x]:
                for dx in (-1, +1):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and 0 <= y + 2 * dy < brd.shape[1] and 0 <= x + 2 * dx < brd.shape[2] \
                        and brd[1 - wrt, y + dy, x + dx] \
                        and not (brd[1 - wrt, y + 2 * dy, x + 2 * dx] | brd[wrt, y + 2 * dy, x + 2 * dx]):
                        frto[at, 0, 0] = y
                        frto[at, 0, 1] = x
                        frto[at, 1, 0] = y + 2 * dy
                        frto[at, 1, 1] = x + 2 * dx
                        at += 1
    return at


@nb.njit # type: ignore
def steps(wrt: int, brd: Mat, frto: NDArray[number]) -> int:
    at = 0
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[wrt, y, x]:
                for dy, dx in ((vdir(wrt), 0), (0, -1), (0, +1)):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and not brd[wrt, y + dy, x + dx] and not brd[1 - wrt, y + dy, x + dx]:
                            frto[at, 0, 0] = y
                            frto[at, 0, 1] = x
                            frto[at, 1, 0] = y + dy
                            frto[at, 1, 1] = x + dx
                            at += 1
    return at


@nb.njit # type: ignore
def branches(wrt: int, brd: Mat) -> tuple[int, NDArray[number]]:
    frto = np.empty((np.sum(brd[wrt]) * 3, 2, 2), dtype=number)
    cnt = capts(wrt, brd, frto)
    if cnt <= 0:
        cnt = steps(wrt, brd, frto)
    return cnt, frto


@nb.njit # type: ignore
def is_end(end: Mat, brd: Mat) -> bool | np.bool_:
    return win(0, end, brd) or win(1, end, brd)


@nb.njit # type: ignore
def standardize_state(for_player: int, board_state: Mat) -> NDArray[Any]:
    output = board_state[for_player] - board_state[1 - for_player]
    if for_player == 0:
        output = output[::-1]
    half = output.shape[1] // 2
    left_weight = np.sum(output[:, :half] != 0)
    right_weight = np.sum(output[:, half:] != 0)
    if left_weight > right_weight:
        output = output[:, ::-1]
    return output


@nb.njit # type: ignore
def behind_last_row(board: NDArray[Any]) -> bool:
    self_last_row = 0
    other_last_row = 0
    for y in range(board.shape[0]):
        if np.any(board[y, :] > 0):
            self_last_row = y
        if np.any(board[y, :] < 0):
            other_last_row = y
    if self_last_row <= other_last_row:
        return True
    return False


@nb.njit # type: ignore
def is_safe(y: int, x: int, board: NDArray[Any]) -> bool:
    sides = 0
    for dx in (-1, +1):
        if (0 <= y + 1 <= board.shape[0]) and (0 <= x + dx <= board.shape[1]):
            if board[y + 1, x + dx] != 0:
                sides += 1
        else:
            sides += 1
    if sides == 2:
        return True
    return False


@nb.njit # type: ignore
def count_safe(board: NDArray[Any]) -> int:
    count = 0
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y, x] == 1:
                count += is_safe(y, x, board)
    return count


@nb.njit # type: ignore
def forward(board: NDArray[Any]) -> int:
    total = 0
    for y in range(board.shape[0]):
        for x in range(board.shape[1]):
            if board[y, x] > 0:
                total += 9 - y
    return total


@nb.njit # type: ignore
def evaluate(for_player: int, end_state: Mat, board_state: Mat) -> float:
    for player in (0, 1):
        if win(player, end_state, board_state):
            if player == for_player:
                return 10000.
            return -10000.
    polar = standardize_state(for_player, board_state)
    if behind_last_row(polar):
        return -10000.
    return 50. * polar.sum() + count_safe(polar) + forward(polar)


@nb.njit # type: ignore
def make_or_unmake(is_capture: np.bool_, unmake: bool, for_player: int, move_from: YX, move_to: YX, board_state: Mat) -> None:
    if is_capture:
        through = (move_from + move_to) // 2
        board_state[1 - for_player, through[0], through[1]] = unmake
    board_state[for_player, move_from[0], move_from[1]] = unmake
    board_state[for_player, move_to[0], move_to[1]] = not unmake


@nb.njit # type: ignore
def is_legal(num_moves: int, move_from: YX, move_to: YX, moves: NDArray[number]) -> int:
    for i in range(num_moves):
        if np.all(move_from == moves[i, 0]) and np.all(move_to == moves[i, 1]):
            return i
    return -1


@nb.njit # type: ignore
def ab_search(is_root: bool, for_player: int, depth: int, alpha: float, beta: float, state: SearchState) -> float:
    state.nodes += 1

    num_moves, moves = branches(for_player, state.board_state)
    if is_root and num_moves <= 1:
        state.move_from = moves[0, 0]
        state.move_to = moves[0, 1]
        return nan

    tt_hash = encode_board(for_player, state.board_state, state.r_ply, state.r_mat)
    tt_idx = tt_hash % state.table()[0].shape[0]
    tt_ent = read_tt(tt_idx, state.table())
    legal = is_legal(num_moves, tt_ent[FROM], tt_ent[TO], moves)
    tt_hit = tt_ent[LOCK] == tt_hash and tt_ent[PLAYER] == for_player and legal >= 0
    state.hits += tt_hit

    if tt_hit and tt_ent[HEIGHT] >= depth:
        if tt_ent[FLAG] == FINAL:
            if is_root:
                state.move_from = tt_ent[FROM]
                state.move_to = tt_ent[TO]
            return tt_ent[SCORE] # type: ignore
        if tt_ent[FLAG] == LBOUND:
            alpha = max(alpha, tt_ent[SCORE]) # type: ignore
        if tt_ent[FLAG] == UBOUND:
            beta = min(beta, tt_ent[SCORE]) # type: ignore
        if alpha >= beta:
            if is_root:
                state.move_from = tt_ent[FROM]
                state.move_to = tt_ent[TO]
            return tt_ent[SCORE] # type: ignore
        
    if depth <= 0 or is_end(state.end_state, state.board_state):
        return evaluate(for_player, state.end_state, state.board_state)
    
    from_best = tt_ent[FROM]; to_best = tt_ent[TO]
    if tt_hit and tt_ent[HEIGHT] >= 0:
        is_capture = is_capt(for_player, from_best, to_best, state.board_state)
        make_or_unmake(is_capture, False, for_player, from_best, to_best, state.board_state)
        decrease = 0 if num_moves <= 1 or (depth <= 1 and is_capture) else 1
        score = -ab_search(False, 1 - for_player, depth - decrease, -beta, -alpha, state)
        make_or_unmake(is_capture, True, for_player, from_best, to_best, state.board_state)

        if score >= beta:
            flag = FINAL
            if score <= alpha:
                flag = UBOUND
            if score >= beta:
                flag = LBOUND
            if tt_ent[HEIGHT] <= depth or tt_ent[AGE] <= state.age:
                state.writes += 1
                write_tt(tt_idx, tt_hash, from_best, to_best, score, flag, depth, age, for_player, state.table()) # type: ignore
            if is_root:
                state.move_from = from_best
                state.move_to = to_best
            return score

    score = -inf
    for i in range(num_moves):
        if not tt_hit or legal != i:
            move_from = moves[i, 0]; move_to = moves[i, 1]
            is_capture = is_capt(for_player, move_from, move_to, state.board_state)
            make_or_unmake(is_capture, False, for_player, move_from, move_to, state.board_state)
            decrease = 0 if num_moves <= 1 or (depth <= 1 and is_capture) else 1
            value = -ab_search(False, 1 - for_player, depth - decrease, -beta, -max(alpha, score), state)
            make_or_unmake(is_capture, True, for_player, move_from, move_to, state.board_state)

            if value > score:
                score = value

                if is_root:
                    state.move_from = move_from
                    state.move_to = move_to

                if score >= beta:
                    flag = FINAL
                    if score <= alpha:
                        flag = UBOUND
                    if score >= beta:
                        flag = LBOUND
                    if tt_ent[HEIGHT] <= depth or tt_ent[AGE] < state.age:
                        state.writes += 1
                        write_tt(tt_idx, tt_hash, move_from, move_to, score, flag, depth, age, for_player, state.table()) # type: ignore
                    break

    return score


@dbg
@nb.njit # type: ignore
def from_root(for_player: int, depth: int, state: SearchState) -> float:
    return ab_search(True, for_player, depth, -inf, +inf, state)


@dataclass
class Stats:
    nodes: int = field(default_factory=lambda: -1)
    hits: int = field(default_factory=lambda: -1)
    writes: int = field(default_factory=lambda: -1)
    age: int = field(default_factory=lambda: -1)
    depth: int = field(default_factory=lambda: -1)


def simple(game: Engine, max_depth: int = 3) -> tuple[float, tuple[YX, YX], Stats]:
    assert table is not None
    assert r_ply is not None
    assert r_mat is not None
    global age
    game = deepcopy(game)
    state = SearchState(np.empty((2,), dtype=np.int8), np.empty((2,), dtype=np.int8), 
                        game.end, game.brd, 
                        r_ply, r_mat, table,
                        age)
    score = nan
    if game.wins is None:
        score = from_root(game.wrt, max_depth, state)
    age += 1
    return score, (state.move_from, state.move_to), Stats(int(state.nodes), int(state.hits), int(state.writes), int(state.age))
    

# NOTE: This is not working properly.
def iterative_deepening(game: Engine, max_time: float, max_depth: int) -> tuple[float, tuple[YX, YX], Stats]:
    assert table is not None
    assert r_ply is not None
    assert r_mat is not None
    global age
    game = deepcopy(game)
    state = SearchState(np.empty((2,), dtype=np.int8), np.empty((2,), dtype=np.int8), 
                        game.end, game.brd, 
                        r_ply, r_mat, table,
                        age)
    
    # Initialize with a random legal move
    num_moves, moves = branches(game.wrt, game.brd)
    if num_moves > 0:
        random_index = np.random.randint(0, num_moves)
        move = (moves[random_index, 0].copy(), moves[random_index, 1].copy())
    else:
        move = (np.empty((2,), dtype=np.int8), np.empty((2,), dtype=np.int8))
    
    score = nan
    start_time = time.time()
    last_depth = -1
    
    for depth in range(1, max_depth + 1):
        if time.time() - start_time >= max_time:
            break

        last_depth = depth
        
        if game.wins is None:
            score = from_root(game.wrt, depth, state)
            move = (state.move_from.copy(), state.move_to.copy())
        
        # Check if we've found a winning move using np.isclose
        if np.isclose(abs(score), 100, atol=1e-6):
            break
    
    age += 1
    return score, move, Stats(int(state.nodes), int(state.hits), int(state.writes), int(state.age), last_depth)

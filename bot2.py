
from collections.abc import Callable
from copy import deepcopy
from typing import Any
from fianco import number, Mat, YX, is_capt, win, vdir, Engine, pieces
from numpy.typing import NDArray
from numpy import inf, nan
from numba.experimental import jitclass # type: ignore

import numpy as np
import numba as nb # type: ignore



def dbg(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrap(*args: Any, **kwargs: Any) -> Any:
        org = set(func.signatures) # type: ignore
        ret = func(*args, **kwargs)
        new = set(func.signatures) # type: ignore
        if new != org:
            print(f'compiled `{func.__name__}` for `{new}`')
        return ret
    return wrap


@jitclass([('end_state', nb.bool[:, :, :]),
           ('board_state', nb.bool[:, :, :]),
           ('move_from', nb.int8[:]),
           ('move_to', nb.int8[:])]) # type: ignore
class SearchState:
    end_state: Mat
    board_state: Mat
    
    move_from: NDArray[number]
    move_to: NDArray[number]

    def __init__(self, 
                 move_from: YX, move_to: YX,
                 end_state: Mat, board_state: Mat) -> None:
        self.end_state = end_state
        self.board_state = board_state

        self.move_from = move_from
        self.move_to = move_to


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
def evaluate(for_player: int, end_state: Mat, board_state: Mat) -> float:
    for player in (0, 1):
        if win(player, end_state, board_state):
            if player == for_player:
                return 1.
            return -1.
    polar = standardize_state(for_player, board_state)
    return polar.sum() / pieces


@nb.njit # type: ignore
def make_or_unmake(is_capture: np.bool_, unmake: bool, for_player: int, move_from: YX, move_to: YX, board_state: Mat) -> None:
    if is_capture:
        through = (move_from + move_to) // 2
        board_state[1 - for_player, through[0], through[1]] = unmake
    board_state[for_player, move_from[0], move_from[1]] = unmake
    board_state[for_player, move_to[0], move_to[1]] = not unmake


@nb.njit # type: ignore
def search(is_root: bool, for_player: int, depth: int, alpha: float, beta: float, state: SearchState) -> float:
    if depth <= 0 or is_end(state.end_state, state.board_state):
        return evaluate(for_player, state.end_state, state.board_state)
    score = -inf
    num_moves, moves = branches(for_player, state.board_state)
    for i in range(num_moves):
        move_from = moves[i, 0]; move_to = moves[i, 1]
        if is_root and num_moves <= 1:
            state.move_from = move_from
            state.move_to = move_to
            return nan
        is_capture = is_capt(for_player, move_from, move_to, state.board_state)
        make_or_unmake(is_capture, False, for_player, move_from, move_to, state.board_state)
        decrease = 0 if num_moves <= 1 else 1
        value = -search(False, 1 - for_player, depth - decrease, -beta, -max(alpha, score), state)
        make_or_unmake(is_capture, True, for_player, move_from, move_to, state.board_state)
        if value > score:
            score = value
            if is_root:
                state.move_from = move_from
                state.move_to = move_to
        if score >= beta:
            break
    return score


@dbg
@nb.njit # type: ignore
def from_root(for_player: int, depth: int, state: SearchState) -> float:
    return search(True, for_player, depth, -inf, +inf, state)


def think(game: Engine, max_depth: int = 3) -> tuple[float, tuple[YX, YX]]:
    game = deepcopy(game)
    state = SearchState(np.empty((2,), dtype=np.int8), np.empty((2,), dtype=np.int8), game.end, game.brd)
    score = nan
    if game.wins is None:
        score = from_root(game.wrt, max_depth, state)
    return score, (state.move_from, state.move_to)
    
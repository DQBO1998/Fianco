from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from numpy import typing as npy
from typing import TypeAlias

import numpy as np
import numba as nb # type: ignore


number: TypeAlias = np.int8
cell: TypeAlias = np.bool_
Mat: TypeAlias = npy.NDArray[cell]
YX: TypeAlias = npy.NDArray[number]


brd_black_init: Mat = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=cell)
brd_white_init: Mat = np.flip(brd_black_init, axis=0)
brd_init: Mat = np.stack((np.copy(brd_black_init), 
                          np.copy(brd_white_init)), axis=0)
end_black_init: Mat = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
], dtype=cell)
end_white_init: Mat = np.flip(end_black_init, axis=0)
end_init: Mat = np.stack((np.copy(end_black_init), 
                          np.copy(end_white_init)), axis=0)
max_time: float = 10.
pieces = 15


def blit(y: int, x: int, fr: Mat, to: Mat) -> None:
    h, w = fr.shape
    to[y:h + y, x:w + x] = fr


def make_board() -> Mat:
    return np.zeros((2, 9, 9), dtype=cell)


def move(wrt: int, mov: Mat, brd: Mat) -> None:
    assert np.sum(mov) == 2, f'only `from` and `to` selections allowed (only 2) - got {np.sum(mov)}\n\n{mov}\n'
    assert np.sum(brd[wrt] & mov) == 1, f'one and only one selection must match - {np.sum(brd[wrt] & mov)} matched\n\n{mov}\n'
    assert np.sum(brd[1 - wrt] & mov) == 0, f'cannot overlap with other player - {np.sum(brd[1 - wrt] & mov)} overlaps\n\n{mov}\n'
    brd[wrt] ^= mov


def capt(wrt: int, msk: Mat, brd: Mat) -> None:
    assert np.sum(msk) == 1, f'must capture one piece - tried {np.sum(msk)}\n\n{msk}\n'
    assert np.sum(brd[wrt] & msk) == 0, f'cannot capture own pieces - tried {np.sum(brd[wrt] & msk)}\n\n{msk}\n'
    assert np.sum(brd[1 - wrt] & msk) == 1, f'must capture from other player - tried {np.sum(brd[1 - wrt] & msk)}\n\n{msk}\n'
    brd[1 - wrt] ^= msk


@nb.njit # type: ignore
def vdir(wrt: int) -> int:
    return 1 - 2 * wrt


@nb.njit # type: ignore
def can_capt(wrt: int, brd: Mat) -> bool:
    dy = vdir(wrt)
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[wrt, y, x]:
                for dx in (-1, +1):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and 0 <= y + 2 * dy < brd.shape[1] and 0 <= x + 2 * dx < brd.shape[2] \
                        and brd[1 - wrt, y + dy, x + dx] \
                        and not (brd[1 - wrt, y + 2 * dy, x + 2 * dx] | brd[wrt, y + 2 * dy, x + 2 * dx]):
                        return True
    return False


@nb.njit # type: ignore
def can_step(wrt: int, brd: Mat) -> bool:
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[wrt, y, x]:
                for dy, dx in ((vdir(wrt), 0), (0, -1), (0, +1)):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and not brd[wrt, y + dy, x + dx] and not brd[1 - wrt, y + dy, x + dx]:
                            return True
    return False


@nb.njit # type: ignore
def is_capt(wrt: int, fr: YX, to: YX, brd: Mat) -> bool | cell:
    fr_y, fr_x = fr
    to_y, to_x = to
    mi = fr + (to - fr) // 2
    mi_y, mi_x = mi
    dx = to_x - fr_x
    dy = to_y - fr_y
    return brd[1 - wrt, mi_y, mi_x] \
           and brd[wrt, fr_y, fr_x] \
           and np.abs(dx) == 2 \
           and np.all(~brd[:, to_y, to_x]) \
           and dy == 2 * vdir(wrt)


@nb.njit # type: ignore
def is_step(wrt: int, fr: YX, to: YX, brd: Mat) -> bool | cell:
    fr_y, fr_x = fr
    to_y, to_x = to
    dx = to_x - fr_x
    dy = to_y - fr_y
    return brd[wrt, fr_y, fr_x] \
           and ((np.abs(dx) > 0) ^ (np.abs(dy) > 0)) \
           and np.all(~brd[:, to_y, to_x]) \
           and (np.abs(dx) == 1 or dy == vdir(wrt))


def inbound(yx: YX, lmt_y: int, lmt_x: int) -> bool:
    y, x = yx
    return 0 <= y < lmt_y and 0 <= x < lmt_x


def fr_to_msk(fr: YX, to: YX, brd: Mat) -> Mat:
    out = np.zeros(brd[0].shape, dtype=brd.dtype)
    fr_y, fr_x = fr
    out[fr_y, fr_x] = True
    to_y, to_x = to
    out[to_y, to_x] = True
    return out


def at_msk(at: YX, brd: Mat) -> Mat:
    out = np.zeros(brd[0].shape, dtype=brd.dtype)
    at_y, at_x = at
    out[at_y, at_x] = True
    return out


def play(in_place: bool, wrt: int, fr: YX, to: YX, brd: Mat) -> tuple[bool, Mat]:
    if inbound(fr, *brd.shape[1:]) and inbound(to, *brd.shape[1:]) \
        and brd[wrt, fr[0], fr[1]] and not brd[wrt, to[0], to[1]] and not brd[1 - wrt, to[0], to[1]]:
        brd = brd if in_place else np.copy(brd)
        if can_capt(wrt, brd):
            if is_capt(wrt, fr, to, brd):
                at = fr + (to - fr) // 2
                capt(wrt, at_msk(at, brd), brd)
                # NOTE: The fact that the line bellow repeats twice makes me crazy.
                # Sometime I will fix this disgrace!
                move(wrt, fr_to_msk(fr, to, brd), brd)
                return True, brd
        elif is_step(wrt, fr, to, brd):
            # NOTE: Yeah, this line is the same as above. Emotional damage!
            move(wrt, fr_to_msk(fr, to, brd), brd)
            return True, brd
    return False, brd


@nb.njit # type: ignore
def win(wrt: int, end: Mat, brd: Mat) -> cell | bool:
    return np.any(brd[wrt] & end[1 - wrt]) \
           or np.all(~brd[1 - wrt]) \
           or (not can_step(1 - wrt, brd) and not can_capt(1 - wrt, brd))


class End(Enum):
    BLACK_WINS = 0
    WHITE_WINS = 1
    TIE = 2


@dataclass
class Engine:
    wrt: int = field(default_factory=lambda: 1)
    brd: Mat = field(default_factory=lambda: np.copy(brd_init))
    end: Mat = field(default_factory=lambda: np.copy(end_init))
    hst: deque[Mat] = field(default_factory=deque)

    @property
    def wins(self) -> End | None:
        # check if black or white wins
        if win(0, self.end, self.brd):
            return End.BLACK_WINS
        if win(1, self.end, self.brd):
            return End.WHITE_WINS
        return None
    
    def play(self, fr: YX, to: YX) -> bool:
        if self.wins is None: # if the game is still going
            ok, nxt = play(False, self.wrt, fr, to, self.brd) # try moving a piece from `fr` to `to`
            if ok: # if the move was successful
                self.wrt = 1 - self.wrt # swap players
                self.hst.append(self.brd) # store the previous state
                self.brd = nxt # address new state
                return True # return `True` to signal the move was successful
        return False # return `False` to signal the move failed

    def undo(self) -> bool:
        if self.hst: # if there are previous moves
            self.wrt = 1 - self.wrt # swap players
            self.brd = self.hst.pop() # address previous move
            return True # return `True` to signal the undo was successful
        return False # return `False` to singal the undo failed

from collections import deque
from dataclasses import dataclass, field
from itertools import product
from numpy import typing as npy
from typing import TypeAlias
from PIL import Image as Img

import numpy as np


Mat: TypeAlias = npy.NDArray[np.bool_]
YX: TypeAlias = npy.NDArray[np.int8]


def blit(y: int, x: int, fr: Mat, to: Mat) -> None:
    h, w = fr.shape
    to[y:h + y, x:w + x] = fr


def make_board() -> Mat:
    return np.zeros((2, 9, 9), dtype=np.bool_)


def move(ply: int, mov: Mat, brd: Mat) -> None:
    assert np.sum(mov) == 2, f'only `from` and `to` selections allowed (only 2) - got {np.sum(mov)}\n\n{mov}\n'
    assert np.sum(brd[ply] & mov) == 1, f'one and only one selection must match - {np.sum(brd[ply] & mov)} matched\n\n{mov}\n'
    assert np.sum(brd[1 - ply] & mov) == 0, f'cannot overlap with other player - {np.sum(brd[1 - ply] & mov)} overlaps\n\n{mov}\n'
    brd[ply] ^= mov


def capt(ply: int, msk: Mat, brd: Mat) -> None:
    assert np.sum(msk) == 1, f'must capture one piece - tried {np.sum(msk)}\n\n{msk}\n'
    assert np.sum(brd[ply] & msk) == 0, f'cannot capture own pieces - tried {np.sum(brd[ply] & msk)}\n\n{msk}\n'
    assert np.sum(brd[1 - ply] & msk) == 1, f'must capture from other player - tried {np.sum(brd[1 - ply] & msk)}\n\n{msk}\n'
    brd[1 - ply] ^= msk


def vdir(ply: int) -> int:
    return 1 - 2 * ply


def can_capt(ply: int, brd: Mat) -> bool:
    K = 2
    h, w = brd.shape[1:]
    slf = np.empty((h + 2 * K, w + 2 * K), dtype=brd.dtype)
    lmt0 = np.ones((h + 2 * K, w + 2 * K), dtype=brd.dtype)
    lmt1 = np.copy(lmt0)
    blit(K, K, brd[1 - ply], lmt0)
    blit(K, K, brd[ply], lmt1)
    lmt = lmt0 | lmt1
    oth = np.zeros((h + 2 * K, w + 2 * K), dtype=brd.dtype)
    blit(K, K, brd[1 - ply], oth)
    dy = vdir(ply)
    for dx in (-1, +1):
        slf.fill(0)
        blit(K + dy, K + dx, brd[ply], slf)
        cap = (slf & oth)[K:-K, K:-K]
        if np.any(cap):
            slf.fill(0)
            blit(K + dy, K + dx, cap, slf)
            lnd = (slf & ~lmt)[K:-K, K:-K]
            if np.any(lnd):
                return True
    return False


def can_step(ply: int, brd: Mat) -> bool:
    K = 1
    h, w = brd.shape[1:]
    slf = np.empty((h + 2 * K, w + 2 * K), dtype=brd.dtype)
    lmt0 = np.ones((h + 2 * K, w + 2 * K), dtype=brd.dtype)
    lmt1 = np.copy(lmt0)
    blit(K, K, brd[1 - ply], lmt0)
    blit(K, K, brd[ply], lmt1)
    lmt = lmt0 | lmt1
    '''oth = np.zeros((h + 2 * K, w + 2 * K), dtype=brd.dtype)
    blit(K, K, brd[1 - ply], oth)'''
    for dy, dx in [(0, -1), (vdir(ply), 0), (0, +1)]:
        slf.fill(0)
        blit(K + dy, K + dx, brd[ply], slf)
        lnd = (slf & ~lmt)[K:-K, K:-K]
        if np.any(lnd):
            return True
    return False


def is_capt(ply: int, fr: YX, to: YX, brd: Mat) -> bool:
    fr_y, fr_x = fr
    to_y, to_x = to
    mi = fr + (to - fr) // 2
    mi_y, mi_x = mi
    dx = to_x - fr_x
    dy = to_y - fr_y
    return all((abs(dx) == 2,
                dy == 2 * vdir(ply),
                brd[ply, fr_y, fr_x],
                np.all(~brd[:, to_y, to_x]),
                brd[1 - ply, mi_y, mi_x]))


def is_step(ply: int, fr: YX, to: YX, brd: Mat) -> bool:
    fr_y, fr_x = fr
    to_y, to_x = to
    dx = to_x - fr_x
    dy = to_y - fr_y
    return all(((abs(dx) > 0) ^ (abs(dy) > 0),
                abs(dx) == 1 or dy == vdir(ply),
                brd[ply, fr_y, fr_x],
                np.all(~brd[:, to_y, to_x])))


def inbound(yx: YX, lmt: tuple[int, ...]) -> bool:
    y, x = yx
    lmt_y, lmt_x = lmt
    return 0 <= y < lmt_y and 0 <= x < lmt_x


def to_msk(fr: YX, to: YX | None, brd: Mat) -> Mat:
    out = np.zeros_like(brd[0])
    fr_y, fr_x = fr
    out[fr_y, fr_x] = True
    if to is not None:
        to_y, to_x = to
        out[to_y, to_x] = True
    return out


def play(ply: int, fr: YX, to: YX, brd: Mat, in_place: bool = False) -> tuple[bool, Mat]:
    if all((np.sum(brd[ply] & brd[1 - ply]) == 0, inbound(fr, brd.shape[1:]), inbound(to, brd.shape[1:]))):
        brd = brd if in_place else np.copy(brd)
        if can_capt(ply, brd):
            if is_capt(ply, fr, to, brd):
                at = fr + (to - fr) // 2
                capt(ply, to_msk(at, None, brd), brd)
                # NOTE: The fact that the line bellow repeats twice makes me crazy.
                # Sometime I will fix this disgrace!
                move(ply, to_msk(fr, to, brd), brd)
                return True, brd
        elif is_step(ply, fr, to, brd):
            # NOTE: Yeah, this line is the same as above. Emotional damage!
            move(ply, to_msk(fr, to, brd), brd)
            return True, brd
    return False, brd


def win(ply: int, end: Mat, brd: Mat) -> bool:
    goal = lambda: np.any(brd[ply] & end[1 - ply])
    dead = lambda: np.all(~brd[1 - ply])
    cant = lambda: not can_step(1 - ply, brd) and not can_capt(1 - ply, brd)
    if goal() or dead() or cant():
        return True
    return False


def load(pth: str) -> Mat:
    brd = make_board()
    with Img.open(pth) as img:
        BLACK = (0,) * 3
        WHITE = (255,) * 3
        image = img.convert('RGB')
        wth, hgt = image.size
        for i, j in product(range(hgt), range(wth)):
            pix = image.getpixel((j, i))
            if pix == BLACK or pix == WHITE:
                brd[int(pix == WHITE), i, j] = True
    return brd


@dataclass
class Engine:
    ply: int = field(default_factory=lambda: 1)
    brd: Mat = field(default_factory=lambda: load(r'D:\Github\Fianco\brd.png'))
    end: Mat = field(default_factory=lambda: load(r'D:\Github\Fianco\end.png'))
    in_place: bool = True
    hst: deque[tuple[YX, YX | None, YX]] = deque()

    @property
    def winner(self) -> int | None:
        for i in (0, 1):
            if win(i, self.end, self.brd):
                return i
        return None
    
    def play(self, fr: YX, to: YX) -> bool:
        if self.winner is None:
            cpt = is_capt(self.ply, fr, to, self.brd)
            ok, nxt = play(self.ply, fr, to, self.brd, self.in_place)
            if ok:
                self.ply = 1 - self.ply
                self.brd = nxt
                self.hst.append((fr, (fr + to) // 2 if cpt else None, to))
                return True
        return False

    def undo(self) -> bool:
        if self.hst:
            self.ply = 1 - self.ply
            (fr_y, fr_x), th, (to_y, to_x) = self.hst.pop()
            if th is not None:
                th_y, th_x = th
                self.brd[1 - self.ply, th_y, th_x] = True
            self.brd[self.ply, fr_y, fr_x] = True
            self.brd[self.ply, to_y, to_x] = False
            return True
        return False
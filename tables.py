from typing import Any, TypeAlias
from numpy.typing import NDArray
from numpy.random import default_rng, Generator
from numba.experimental import jitclass # type: ignore
from fianco import YX, Mat, number, is_step, is_capt, can_capt, cell, brd_init
from sys import getsizeof as sizeof

import numpy as np
import numba as nb # type: ignore


Key: TypeAlias = np.uint64

@jitclass([('hgt', nb.int32),
           ('flg', nb.uint8),
           ('scr', nb.float32),
           ('fr_y', nb.uint8), ('fr_x', nb.uint8),
           ('to_y', nb.uint8), ('to_x', nb.uint8),
           ('lck', nb.uint64)]) # type: ignore
class Entry:
    hgt: int
    flg: int
    scr: float
    fr_y: int
    fr_x: int
    to_y: int
    to_x: int
    lck: int

    def __init__(self, hgt: int,
                       flg: int,
                       scr: float,
                       fr_y: int,
                       fr_x: int,
                       to_y: int,
                       to_x: int,
                       lck: int) -> None:
        self.hgt = hgt
        self.flg = flg
        self.scr = scr
        self.fr_y = fr_y; self.fr_x = fr_x
        self.to_y = to_y; self.to_x = to_x
        self.lck = lck


RandomNumbers: TypeAlias = NDArray[Key]
Table: TypeAlias = list[Entry]


LB = 0
UB = 1
VL = 2

HGT = 0
FLG = HGT + 1
SCR = FLG + 1
MOV = SCR + 1
LCK = MOV + 1


@nb.njit # type: ignore
def is_legal(wrt: int, fr: YX, to: YX, brd: Mat) -> bool | cell:
    return (can_capt(wrt, brd) and is_capt(wrt, fr, to, brd)) or (is_step(wrt, fr, to, brd) and not can_capt(wrt, brd))


@nb.njit # type: ignore
def std_brd(wrt: int, brd: Mat, flip: bool) -> tuple[NDArray[Any], bool, bool]:
    vflip = False
    hflip = False
    out = brd[wrt] - brd[1 - wrt]
    if flip and wrt == 0:
        vflip = True
        out = out[::-1]
    at = out.shape[1] // 2
    lw = np.sum(out[:, :at] != 0)
    rw = np.sum(out[:, at:] != 0)
    if flip and lw > rw:
        hflip = True
        out = out[:, ::-1]
    return out, vflip, hflip


@nb.njit # type: ignore
def flip_mov(vflip: bool, hflip: bool, y_dim: int, x_dim: int, mov: NDArray[number]) -> NDArray[number]:
    if vflip:
        mov[:, 0] = y_dim - mov[:, 0] - 1
    if hflip:
        mov[:, 1] = x_dim - mov[:, 1] - 1
    return mov


@nb.njit # type: ignore
def enc_brd(wrt: int, brd: Mat, r_mat: RandomNumbers) -> tuple[int, bool, bool]:
    brd, vflip, hflip = std_brd(wrt, brd, False)
    hsh = 0
    for y in range(brd.shape[0]):
        for x in range(brd.shape[1]):
            if brd[y, x] != 0:
                hsh ^= r_mat[brd[y, x], y, x]
    return hsh, vflip, hflip


@nb.njit # type: ignore
def read_tt(wrt: int, brd: Mat, r_mat: RandomNumbers, tt: Table) -> tuple[bool | cell, int, tuple[bool, bool], NDArray[number]]:
    hsh, vflip, hflip = enc_brd(wrt, brd, r_mat)
    idx = hsh % len(tt)
    tt_mov = np.empty((2, 2), dtype=number)
    tt_mov[0, 0] = tt[idx].fr_y
    tt_mov[0, 1] = tt[idx].fr_x
    tt_mov[1, 0] = tt[idx].to_y
    tt_mov[1, 1] = tt[idx].to_x
    tt_mov = flip_mov(vflip, hflip, brd.shape[1], brd.shape[2], tt_mov)
    fr = tt_mov[0]
    to = tt_mov[1]
    ok = hsh == tt[idx].lck and is_legal(wrt, fr, to, brd)
    return ok, idx, (vflip, hflip), tt_mov


@nb.njit # type: ignore
def write_tt(vflip: bool, hflip: bool, y_dim: int, x_dim: int, 
             hsh: int, hgt: int, flg: int, scr: float, mov: NDArray[number], 
             tt: Table) -> None:
    idx = hsh % len(tt)
    tt[idx].hgt = hgt
    tt[idx].flg = flg
    tt[idx].scr = scr
    frto = flip_mov(vflip, hflip, y_dim, x_dim, np.copy(mov))
    tt[idx].fr_y = frto[0, 0]
    tt[idx].fr_x = frto[0, 1]
    tt[idx].to_y = frto[1, 0]
    tt[idx].to_x = frto[1, 1]
    tt[idx].lck = hsh


def make_tt(size: int, rnd: Generator) -> tuple[RandomNumbers, Table]:
    tt = [Entry(-1, LB, -np.inf, 0, 0, 0, 0, 0) for _ in range(size)]
    r_mat = rnd.integers(0, np.iinfo(Key).max, (2, 9, 9), dtype=Key)
    return r_mat, tt


if __name__ == '__main__':
    r_mat, tt = make_tt(10000, default_rng(3423))
    print(sizeof(tt))
    brd = np.copy(brd_init)
    std, _, _ = std_brd(1, brd, True)
    print(std)
    hsh, vf, hf = enc_brd(1, brd, r_mat)
    print(hsh, vf, hf)
    mov = np.array([(8, 0), (7, 0)], dtype=number)
    write_tt(False, False, 9, 9, hsh, 0, LB, -np.inf, mov, tt)
    print(read_tt(1, brd, r_mat, tt))
    print(tt[hsh % len(tt)].hgt)


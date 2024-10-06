
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast
from fianco import number, Mat, YX, is_capt, can_capt, win, vdir, Engine, pieces
from numpy.typing import NDArray
from numpy.random import default_rng, Generator
from numpy import inf, nan, finfo
from tables import *

import numpy as np
import numba as nb # type: ignore


MAX_SCR: float = float(finfo(np.float16).max - 1)
MIN_SCR: float = float(finfo(np.float16).min + 1)

ROWS_0 = np.array([9 - (i + 1) for i in range(9)], dtype=np.int32)
COLS_0 = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1], dtype=np.int32)

NDC = 0
TT_HIT = 1
TT_ATT = 2


def dbg(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrap(*args: Any, **kwargs: Any) -> Any:
        org = set(func.signatures) # type: ignore
        ret = func(*args, **kwargs)
        new = set(func.signatures) # type: ignore
        if new != org:
            print(f'compiled `{func.__name__}` for `{new}`')
        return ret
    return wrap


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
def brnchs(wrt: int, brd: Mat) -> tuple[int, NDArray[number]]:
    frto = np.empty((np.sum(brd[wrt]) * 3, 2, 2), dtype=number)
    cnt = capts(wrt, brd, frto)
    if cnt <= 0:
        cnt = steps(wrt, brd, frto)
    return cnt, frto


@nb.njit # type: ignore
def is_qui(wrt: int, brd: Mat) -> bool:
    return can_capt(wrt, brd)


@nb.njit # type: ignore
def is_end(end: Mat, brd: Mat) -> bool | np.bool_:
    return win(0, end, brd) or win(1, end, brd)


'''@nb.stencil # type: ignore
def rsame(x: NDArray[Any], y: NDArray[Any]) -> float:
    return (x[0, 0] * y[0,0] + 1) / 2


@nb.njit # type: ignore
def conv2d(brd: NDArray[Any], krn: NDArray[Any]) -> NDArray[Any]:
    out = np.zeros([s + 2 for s in brd.shape], dtype=krn.dtype)
    msk = np.ones(brd.shape, dtype=krn.dtype)
    blit(1, 1, msk, out)
    for y in range(brd.shape[0]):
        for x in range(brd.shape[1]):
            for i in range(krn.shape[0]):
                for j in range(krn.shape[1]):
                    out[y, x] *= (brd[y + i - krn.shape[0] // 2, x + j - krn.shape[1] // 2] * krn[i, j] + 1) / 2
    return out'''


@nb.njit # type: ignore
def scr_at(wrt: int, end: Mat, brd: Mat) -> float:
    for p in (0, 1):
        if win(p, end, brd):
            if p == wrt:
                return MAX_SCR
            return MIN_SCR
    brd, _, _ = std_brd(wrt, brd, True)
    diff = np.sum(brd) / pieces
    return diff # type: ignore


@nb.njit # type: ignore
def do(capt: np.bool_, flip: bool, wrt: int, fr: YX, to: YX, brd: Mat) -> None:
    if capt:
        th = (fr + to) // 2
        brd[1 - wrt, th[0], th[1]] = flip
    brd[wrt, fr[0], fr[1]] = flip
    brd[wrt, to[0], to[1]] = not flip


@nb.njit # type: ignore
def rec_mov(fr: YX, to: YX, mov: NDArray[number]) -> None:
    mov[0, :] = fr
    mov[1, :] = to


@nb.njit # type: ignore
def search(root: bool,
           wrt: int, dpth: int, α: float, β: float, 
           end: Mat, brd: Mat, 
           tt: Table, mov: NDArray[number], stats: NDArray[np.int64]) -> float:
    # count visited nodes
    stats[NDC] += 1
    # read transposition table (tt)
    tt_ok, tt_i, (vflip, hflip), tt_mov = read_tt(wrt, brd, r_mat, tt)
    # count total retrieval attempts and successful attempts
    stats[TT_ATT] += 1
    stats[TT_HIT] += tt_ok
    # try to prune the whole sub-tree
    if tt_ok and tt[tt_i].hgt >= dpth:
        if tt[tt_i].flg == LB:
            α = max(α, tt[tt_i].scr)
        if tt[tt_i].flg == UB:
            β = min(β, tt[tt_i].scr)
        # NOTE: either way, the root node will need the move
        if tt[tt_i].flg == VL or α >= β:
            if root:
                fr = tt_mov[0]
                to = tt_mov[1]
                rec_mov(fr, to, mov)
            return tt[tt_i].scr
    # if max. depth or endgame, evaluate position
    if dpth <= 0 or is_end(end, brd):
        return scr_at(wrt, end, brd)
    # try TT recorded move first, then continue with standard AB-Search
    scr = -inf
    if tt_ok and tt[tt_i].hgt >= 0:
        fr = tt_mov[0]; to = tt_mov[1]
        capt = is_capt(wrt, fr, to, brd)
        do(capt, False, wrt, fr, to, brd) # type: ignore
        scr = -search(False, 1 - wrt, dpth - 1, -β, -α, end, brd, tt, mov, stats)
        do(capt, True, wrt, fr, to, brd) # type: ignore
        if scr >= β:
            
    # count and list all available moves
    cnt, frto = brnchs(wrt, brd)
    # for all available moves - search
    for i in range(cnt):
        fr = frto[i, 0]; to = frto[i, 1]
        if root and cnt <= 1:
            rec_mov(fr, to, mov)
            return nan
        capt = is_capt(wrt, fr, to, brd)
        do(capt, False, wrt, fr, to, brd) # type: ignore
        dec = 0 if cnt <= 1 or (capt and dpth <= 1) else 1
        val = -search(False, 1 - wrt, dpth - dec, -β, -max(α, scr), end, brd, tt, mov, stats)
        do(capt, True, wrt, fr, to, brd) # type: ignore
        if val > scr:
            scr = val
            if root:
                rec_mov(fr, to, mov)
        if scr >= β:
            break
    return scr


@dbg
@nb.njit # type: ignore
def root(wrt: int, dpth: int, α: float, β: float, 
         end: Mat, brd: Mat, 
         tt: Table, mov: NDArray[number], stats: NDArray[np.int64]) -> float:
    return search(True, wrt, dpth, -inf, +inf, end, brd, tt, mov, stats)


@dataclass
class FromBot:
    ...


class Bot:
    rm: NDArray[Key]
    tt: tuple[Any, ...]

    def __init__(self, tt_size: int = 2 ** 64) -> None:
        pass


def think(lst: Engine, dpth: int = 3) -> tuple[float, tuple[YX, YX]]:
    lst = deepcopy(lst)
    out = np.zeros((2, 2), dtype=number)
    exp = root(lst.wrt, dpth, -inf, +inf, lst.end, lst.brd, out)
    return exp, cast(tuple[YX, YX], out)


def monke(lst: Engine, rnd: Generator = default_rng(42)) -> tuple[float, tuple[YX, YX]]:
    cnt, frto = brnchs(lst.wrt, lst.brd)
    idx = rnd.integers(0, cnt, endpoint=False)
    return 0., (frto[idx, 0], frto[idx, 1])
    
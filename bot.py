
from math import inf
from copy import deepcopy
from fianco import number, Mat, YX, is_capt, win, vdir, Engine
from numpy.typing import NDArray
from numpy.random import default_rng, Generator

import numpy as np
import numba as nb # type: ignore


@nb.njit # type: ignore
def capts(ply: int, brd: Mat, frto: NDArray[number]) -> int:
    at = 0
    dy = vdir(ply)
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[ply, y, x]:
                for dx in (-1, +1):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and 0 <= y + 2 * dy < brd.shape[1] and 0 <= x + 2 * dx < brd.shape[2] \
                        and brd[1 - ply, y + dy, x + dx] \
                        and not (brd[1 - ply, y + 2 * dy, x + 2 * dx] | brd[ply, y + 2 * dy, x + 2 * dx]):
                        frto[at, 0, 0] = y
                        frto[at, 0, 1] = x
                        frto[at, 1, 0] = y + 2 * dy
                        frto[at, 1, 1] = x + 2 * dx
                        at += 1
    return at


@nb.njit # type: ignore
def steps(ply: int, brd: Mat, frto: NDArray[number]) -> int:
    at = 0
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[ply, y, x]:
                for dy, dx in ((vdir(ply), 0), (0, -1), (0, +1)):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and not brd[ply, y + dy, x + dx] and not brd[1 - ply, y + dy, x + dx]:
                            frto[at, 0, 0] = y
                            frto[at, 0, 1] = x
                            frto[at, 1, 0] = y + dy
                            frto[at, 1, 1] = x + dx
                            at += 1
    return at


@nb.njit # type: ignore
def all_moves(ply: int, brd: Mat) -> tuple[int, NDArray[number]]:
    frto = np.empty((np.sum(brd[ply]) * 3, 2, 2), dtype=number)
    cnt = capts(ply, brd, frto)
    if cnt <= 0:
        cnt = steps(ply, brd, frto)
    return cnt, frto


@nb.njit # type: ignore
def is_end(end: Mat, brd: Mat) -> bool | np.bool_:
    return win(0, end, brd) or win(1, end, brd)


@nb.njit # type: ignore
def scr_at(wrt: int, end: Mat, brd: Mat) -> float:
    w0 = win(0, end, brd)
    if w0 or win(1, end, brd):
        top = 0 if w0 else 1
        return -((top == wrt) - (top != wrt))
    return -((brd[wrt].sum() - brd[1 - wrt].sum()) / 15.)


@nb.njit # type: ignore
def make(capt: bool, flip: bool, ply: int, fr: YX, to: YX, brd: Mat) -> None:
    if capt:
        th = (fr + to) // 2
        brd[1 - ply, th[0], th[1]] = flip
    brd[ply, fr[0], fr[1]] = flip
    brd[ply, to[0], to[1]] = not flip


@nb.njit # type: ignore
def search(root: bool,
           wrt: int, 
           ply: int, dpth: int, α: float, β: float, 
           end: Mat, brd: Mat, 
           ndc: NDArray[np.uint64], out: NDArray[number]) -> float:
    if dpth <= 0 or is_end(end, brd):
        return scr_at(wrt, end, brd)
    scr = -inf
    cnt, frto = all_moves(ply, brd)
    for i in range(cnt):
        ndc[0] += 1
        fr = frto[i, 0]; to = frto[i, 1]
        capt = is_capt(ply, fr, to, brd)
        make(capt, False, ply, fr, to, brd)
        val = -search(False, wrt, 1 - ply, dpth - 1, -β, -max(α, scr), end, brd, ndc, out)
        make(capt, True, ply, fr, to, brd)
        if val > scr:
            scr = val
            if root:
                out[0, :] = fr
                out[1, :] = to
        if scr >= β:
            break
    return scr


def think(lst: Engine, dpth: int = 3) -> tuple[int, float, tuple[YX, YX]]:
    lst = deepcopy(lst)
    ndc = np.zeros((1,), np.uint64)
    out = np.zeros((2, 2), dtype=number)
    exp = search(True, lst.ply, lst.ply, dpth, -inf, +inf, lst.end, lst.brd, ndc, out)
    return ndc[0], exp, out


def monke(lst: Engine, rnd: Generator = default_rng(42)) -> tuple[int, int, tuple[YX, YX]]:
    cnt, frto = all_moves(lst.ply, lst.brd)
    idx = rnd.integers(0, cnt, endpoint=False)
    return 0, 0, (frto[idx, 0], frto[idx, 1])
    
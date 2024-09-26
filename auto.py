
from math import inf
from copy import deepcopy
from fianco import number, Mat, YX, is_capt, win, vdir, move, capt, Engine
from numpy.typing import NDArray

import numpy as np
import numba as nb # type: ignore


@nb.njit # type: ignore
def capts(ply: int, brd: Mat) -> tuple[int, NDArray[number]]:
    at = 0
    top = np.sum(brd[ply]) * 2
    out = np.empty((top, 2, 2), dtype=number)
    dy = vdir(ply)
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[ply, y, x]:
                for dx in (-1, +1):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and 0 <= y + 2 * dy < brd.shape[1] and 0 <= x + 2 * dx < brd.shape[2] \
                        and brd[1 - ply, y + dy, x + dx] \
                        and ~(brd[1 - ply, y + 2 * dy, x + 2 * dx] | brd[ply, y + 2 * dy, x + 2 * dx]):
                        out[at, 0, 0] = y
                        out[at, 0, 1] = x
                        out[at, 1, 0] = y + 2 * dy
                        out[at, 1, 1] = x + 2 * dx
                        at += 1
    return at, out


@nb.njit # type: ignore
def steps(ply: int, brd: Mat) -> tuple[int, NDArray[number]]:
    at = 0
    top = np.sum(brd[ply]) * 3
    out = np.empty((top, 2, 2), dtype=number)
    for y in range(0, brd.shape[1]):
        for x in range(0, brd.shape[2]):
            if brd[ply, y, x]:
                for dy, dx in ((vdir(ply), 0), (0, -1), (0, +1)):
                    if 0 <= y + dy < brd.shape[1] and 0 <= x + dx < brd.shape[2] \
                        and not brd[ply, y + dy, x + dx] and not brd[1 - ply, y + dy, x + dx]:
                            out[at, 0, 0] = y
                            out[at, 0, 1] = x
                            out[at, 1, 0] = y + dy
                            out[at, 1, 1] = x + dx
                            at += 1
    return at, out


@nb.njit # type: ignore
def all_moves(ply: int, brd: Mat) -> tuple[int, NDArray[number]]:
    cnt, frto = capts(ply, brd)
    if cnt <= 0:
        cnt, frto = steps(ply, brd)
    return cnt, frto


@nb.njit # type: ignore
def as_move(ply: int, fr: YX, to: YX, brd: Mat) -> tuple[Mat, Mat]:
    mov = np.zeros(brd.shape[1:], dtype=brd.dtype)
    mov[fr[0], fr[1]] = True
    mov[to[0], to[1]] = True
    cap = np.zeros(brd.shape[1:], dtype=brd.dtype)
    if is_capt(ply, fr, to, brd):
        th = (fr + to) // 2
        cap[th[0], th[1]] = True
    return mov, cap


@nb.njit # type: ignore
def is_end(end: Mat, brd: Mat) -> bool | np.bool_:
    return win(0, end, brd) or win(1, end, brd)


@nb.njit # type: ignore
def scr_at(wrt: int, end: Mat, brd: Mat) -> float:
    w0 = win(0, end, brd)
    if w0 or win(1, end, brd):
        top = 0 if w0 else 1
        return (top == wrt) - (top != wrt)
    return (brd[wrt].sum() - brd[1 - wrt].sum()) / 15.


@nb.njit # type: ignore
def nxtst(ply: int, fr: YX, to: YX, brd: Mat) -> Mat:
    brd = np.copy(brd)
    mov, cap = as_move(ply, fr, to, brd)
    move(ply, mov, brd)
    capt(ply, cap, brd)
    return brd


@nb.njit # type: ignore
def αβ_search(wrt: int, 
              ply: int, dpth: int, α: float, β: float, 
              end: Mat, brd: Mat, 
              _ndc: NDArray[np.uint64]) -> float:
    if dpth <= 0 or is_end(end, brd):
        return scr_at(wrt, end, brd)
    scr = -inf
    cnt, frto = all_moves(ply, brd)
    for i in range(cnt):
        _ndc[0] += 1
        val = -αβ_search(wrt, 1 - ply, dpth - 1, -β, -α, end, nxtst(ply, frto[i, 0], frto[i, 1], brd), _ndc)
        if val > scr:
            scr = val
        if scr > α:
            α = scr
        if scr >= β:
            break
    return scr


def think(lst: Engine, dpth: int = 3, αβ: tuple[float, float] = (-inf, +inf)) -> tuple[int, tuple[YX, YX]]:
    ndc = np.zeros((1,), np.uint64)
    lst = deepcopy(lst)
    α, β = αβ
    out: tuple[YX, YX] | None = None
    scr = -inf
    cnt, frto = all_moves(lst.ply, lst.brd)
    for i in range(cnt):
        ndc[0] += 1
        val = -αβ_search(lst.ply, 1 - lst.ply, dpth - 1, -β, -α, lst.end, nxtst(lst.ply, frto[i, 0], frto[i, 1], lst.brd), ndc)
        if val > scr:
            scr = val
            out = (frto[i, 0], frto[i, 1])
    assert out is not None
    return ndc[0], out
    
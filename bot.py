
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast
from fianco import number, Mat, YX, is_capt, win, vdir, Engine
from numpy.typing import NDArray
from numpy.random import default_rng, Generator
from numpy import inf

import numpy as np
import numba as nb # type: ignore


BIG_M: int = 1000000
ROWS_0 = np.array([i + 1 for i in range(9)], dtype=np.int32)
COLS_0 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)



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
def all_moves(wrt: int, brd: Mat) -> tuple[int, NDArray[number]]:
    frto = np.empty((np.sum(brd[wrt]) * 3, 2, 2), dtype=number)
    cnt = capts(wrt, brd, frto)
    if cnt <= 0:
        cnt = steps(wrt, brd, frto)
    return cnt, frto


@nb.njit # type: ignore
def is_qui(wrt: int, brd: Mat) -> bool:
    return False


@nb.njit # type: ignore
def is_end(end: Mat, brd: Mat) -> bool | np.bool_:
    return win(0, end, brd) or win(1, end, brd)


@nb.njit # type: ignore
def mdiff(wrt: int, brd: Mat) -> float:
    return np.sum(brd[wrt]) - np.sum(brd[1 - wrt])


@nb.njit # type: ignore
def glval(wrt: int, brd: Mat, rows: NDArray[Any], cols: NDArray[Any]) -> float:
    msk = brd[wrt]
    s = 0
    for i in range(rows.shape[0]):
        for j in range(cols.shape[0]):
            at = i
            if wrt == 0:
                at = rows.shape[0] - 1 - i
            s += msk[at, j] * (rows[at] + cols[j])
    return s


@nb.njit # type: ignore
def scr_at(wrt: int, end: Mat, brd: Mat, rows: NDArray[Any], cols: NDArray[Any]) -> float:
    for p in (0, 1):
        if win(p, end, brd):
            if p == wrt:
                return BIG_M
            return -BIG_M
    mtΔ = mdiff(wrt, brd)
    glΔ = glval(wrt, brd, rows, cols) - glval(1 - wrt, brd, rows, cols)
    λ = 20
    return λ * mtΔ + glΔ


@nb.njit # type: ignore
def make(capt: np.bool_, flip: bool, wrt: int, fr: YX, to: YX, brd: Mat) -> None:
    if capt:
        th = (fr + to) // 2
        brd[1 - wrt, th[0], th[1]] = flip
    brd[wrt, fr[0], fr[1]] = flip
    brd[wrt, to[0], to[1]] = not flip


@nb.njit # type: ignore
def search(root: bool,
           wrt: int, dpth: int, α: float, β: float, 
           end: Mat, brd: Mat, 
           out: NDArray[number],
           rows: NDArray[Any], cols: NDArray[Any]) -> float:
    if dpth <= 0 or is_end(end, brd):
        return scr_at(wrt, end, brd, rows, cols)
    scr = -inf
    cnt, frto = all_moves(wrt, brd)
    for i in range(cnt):
        fr = frto[i, 0]; to = frto[i, 1]
        capt = is_capt(wrt, fr, to, brd)
        make(capt, False, wrt, fr, to, brd)
        val = -search(False, 1 - wrt, dpth - 1, -β, -max(α, scr), end, brd, out, rows, cols)
        make(capt, True, wrt, fr, to, brd)
        if val > scr:
            scr = val
            if root:
                out[0, :] = fr
                out[1, :] = to
        if scr >= β:
            break
    return scr


@dbg
@nb.njit # type: ignore
def root(wrt: int, dpth: int, α: float, β: float, 
         end: Mat, brd: Mat, 
         out: NDArray[number],
         rows: NDArray[Any], cols: NDArray[Any]) -> float:
    return search(True, wrt, dpth, -inf, +inf, end, brd, out, rows, cols)


def think(lst: Engine, dpth: int = 3) -> tuple[float, tuple[YX, YX]]:
    lst = deepcopy(lst)
    out = np.zeros((2, 2), dtype=number)
    exp = root(lst.wrt, dpth, -inf, +inf, lst.end, lst.brd, out, ROWS_0, COLS_0)
    return exp, cast(tuple[YX, YX], out)


def monke(lst: Engine, rnd: Generator = default_rng(42)) -> tuple[float, tuple[YX, YX]]:
    cnt, frto = all_moves(lst.wrt, lst.brd)
    idx = rnd.integers(0, cnt, endpoint=False)
    return 0., (frto[idx, 0], frto[idx, 1])
    
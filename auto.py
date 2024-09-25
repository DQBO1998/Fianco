
from collections.abc import Generator
from math import inf
from copy import deepcopy
from fianco import *

import numba as nb # type: ignore
import gc


@nb.njit # type: ignore
def capts(ply: int, brd: Mat) -> Generator[tuple[YX, YX], None, None]:
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
                to = np.swapaxes(np.stack(np.nonzero(lnd), axis=0), 0, 1)
                fr = to - 2 * np.array((dy, dx), dtype=number)
                for n in range(to.shape[0]):
                    yield fr[n], to[n]


@nb.njit # type: ignore
def steps(ply: int, brd: Mat) -> Generator[tuple[YX, YX], None, None]:
    if not can_capt(ply, brd):
        K = 1
        h, w = brd.shape[1:]
        slf = np.empty((h + 2 * K, w + 2 * K), dtype=brd.dtype)
        lmt0 = np.ones((h + 2 * K, w + 2 * K), dtype=brd.dtype)
        lmt1 = np.copy(lmt0)
        blit(K, K, brd[1 - ply], lmt0)
        blit(K, K, brd[ply], lmt1)
        lmt = lmt0 | lmt1
        for dy, dx in [(0, -1), (vdir(ply), 0), (0, +1)]:
            slf.fill(0)
            blit(K + dy, K + dx, brd[ply], slf)
            lnd = (slf & ~lmt)[K:-K, K:-K]
            if np.any(lnd):
                to = np.swapaxes(np.stack(np.nonzero(lnd), axis=0), 0, 1)
                fr = to - np.array((dy, dx), dtype=number)
                for n in range(to.shape[0]):
                    yield fr[n], to[n]


@nb.njit # type: ignore
def capts_and_steps(ply: int, brd: Mat) -> Generator[tuple[YX, YX], None, None]:
    for frto in capts(ply, brd):
        yield frto
    for frto in steps(ply, brd):
        yield frto


@nb.njit # type: ignore
def terminal(end: Mat, brd: Mat) -> bool | np.bool_:
    return win(0, end, brd) or win(1, end, brd)


@nb.njit # type: ignore
def evaluate(wrt: int, end: Mat, brd: Mat) -> float:
    w0 = win(0, end, brd)
    if w0 or win(1, end, brd):
        top = 0 if w0 else 1
        return (top == wrt) - (top != wrt)
    return (brd[wrt].sum() - brd[1 - wrt].sum()) / 15.


@nb.njit # type: ignore
def simulate(ply: int, fr: YX, to: YX, brd: Mat) -> Mat:
    #assert not win(0, end, brd) and not win(1, end, brd), f'bro, the game is over'
    #assert np.sum(brd[ply] & brd[1 - ply]) == 0 and inbound(fr, *brd.shape[1:]) and inbound(to, *brd.shape[1:])
    brd = np.copy(brd)
    if can_capt(ply, brd):
        if is_capt(ply, fr, to, brd):
            at = fr + (to - fr) // 2
            capt(ply, at_msk(at, brd), brd)
            # NOTE: The fact that the line bellow repeats twice makes me crazy.
            # Sometime I will fix this disgrace!
            move(ply, fr_to_msk(fr, to, brd), brd)
    elif is_step(ply, fr, to, brd):
        # NOTE: Yeah, this line is the same as above. Emotional damage!
        move(ply, fr_to_msk(fr, to, brd), brd)
    return brd


@nb.njit # type: ignore
def αβ_search(wrt: int, ply: int, dpth: int, α: float, β: float, end: Mat, brd: Mat) -> tuple[int, float]:
    ndc = 0
    if dpth <= 0 or terminal(end, brd):
        return ndc, evaluate(wrt, end, brd)
    scr = -inf
    for (fr, to) in capts_and_steps(ply, brd):
        add, _val = αβ_search(wrt, 1 - ply, dpth - 1, -β, -α, end, simulate(ply, fr, to, brd))
        val = -_val
        ndc += add + 1
        if val > scr:
            scr = val
        if scr > α:
            α = scr
        if scr >= β:
            break
    return ndc, scr


def think(lst: Engine, dpth: int = 3, αβ: tuple[float, float] = (-inf, +inf)) -> tuple[int, tuple[YX, YX] | None]:
    ndc = 0
    lst = deepcopy(lst)
    if terminal(lst.end, lst.brd):
        return ndc, None
    α, β = αβ
    out: tuple[YX, YX] | None = None
    scr = -inf
    for frto in capts_and_steps(lst.ply, lst.brd):
        add, _val = αβ_search(lst.ply, 1 - lst.ply, dpth - 1, -β, -α, lst.end, simulate(lst.ply, *frto, lst.brd))
        val = -_val
        ndc += add + 1
        if val > scr:
            scr = val
            out = frto
    assert out is not None
    #gc.collect()
    return ndc, out
    
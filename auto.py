
from collections.abc import Callable, Iterable
from math import inf
from itertools import chain
from random import Random
from typing import Any
from fianco import *


rnd = Random(42)


State: TypeAlias = tuple[int, Mat, Mat]
Evalf: TypeAlias = Callable[[State, tuple[Any, ...]], float]


def capts(ply: int, brd: Mat) -> Iterable[tuple[YX, YX]]:
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
                to = np.stack(np.nonzero(lnd), axis=0, dtype=np.int8).swapaxes(0, 1)
                fr = to - 2 * np.stack((dy, dx), axis=0, dtype=np.int8)
                for n in range(to.shape[0]):
                    yield fr[n], to[n]


def steps(ply: int, brd: Mat) -> Iterable[tuple[YX, YX]]:
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
                to = np.stack(np.nonzero(lnd), axis=0, dtype=np.int8).swapaxes(0, 1)
                fr = to - np.stack((dy, dx), axis=0, dtype=np.int8)
                for n in range(to.shape[0]):
                    yield fr[n], to[n]


class IllegalMove(Exception):
    pass


def ft2brd(ply: int, brd: Mat, frto: tuple[YX, YX]) -> Mat:
    ok, nxt = play(ply, *frto, brd)
    if not ok:
        raise IllegalMove(f'going from {frto[0]} to {frto[1]} in \n{brd}\nis illegal, resulting in \n{nxt}\n')
    return nxt 


def terminal(lst: tuple[int, Mat, Mat]) -> bool:
    _, end, brd = lst
    return any((win(ply, end, brd) for ply in (0, 1)))


def simulate(lst: tuple[int, Mat, Mat]) -> float:
    ply, end, brd = lst
    org = ply
    while not terminal(lst):
        frto = rnd.choice(tuple(chain(capts(ply, brd), steps(ply, brd))))
        brd = ft2brd(ply, brd, frto)
        ply = 1 - ply
    return win(org, end, brd)


def mcsim(lst: tuple[int, Mat, Mat], n_trials: int) -> float:
    res = np.array([simulate(lst) for _ in range(n_trials)], dtype=np.float16)
    return res.mean()


def mdist(lst: tuple[int, Mat, Mat]) -> float:
    ply, _, brd = lst
    top0 = 
    mat = brd[0]
    if ply == 1:
        mat = np.flip(brd[ply], axis=0)
    ys, _ = np.nonzero(mat)
    return ys.max() / brd.shape[1]


def αβ_search(lst: tuple[int, Mat, Mat], evalf: tuple[Any, ...], dpth: int, αβ: tuple[float, float]) -> float:
    if dpth <= 0 or terminal(lst):
        fn, *args = evalf
        return fn(lst, *args)
    α, β = αβ
    ply, end, brd = lst
    scr = -inf
    for frto in chain(capts(ply, brd), steps(ply, brd)):
        nxt = ft2brd(ply, brd, frto)
        val = -αβ_search((1 - ply, end, nxt), evalf, dpth - 1, (-β, -α))
        if val > scr:
            scr = val
        if scr > α:
            α = scr
        if scr >= β:
            break
    return scr


def mk_nxt(lst: tuple[int, Mat, Mat], 
           evalf: tuple[Any, ...],
           frto: tuple[YX, YX],
           dpth: int, αβ: tuple[float, float]) -> tuple[float, tuple[YX, YX]]:
    α, β = αβ
    ply, end, brd = lst
    nxt = ft2brd(ply, brd, frto)
    val = -αβ_search((1 - ply, end, nxt), evalf, dpth - 1, (-β, -α))
    return val, frto


def decide(lst: tuple[int, Mat, Mat], 
           evalf: tuple[Any, ...] = (mdist,),
           dpth: int = 3, αβ: tuple[float, float] = (-inf, inf)) -> tuple[YX, YX]:
    ply, _, brd = lst
    nxts = (mk_nxt(lst, evalf, frto, dpth, αβ) for frto in chain(capts(ply, brd), steps(ply, brd)))
    _, frto = max(tuple(nxts), key=lambda vn: vn[0])
    return frto
    
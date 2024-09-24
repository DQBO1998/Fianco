
from collections.abc import Iterable
from math import inf
from itertools import chain
from typing import Union
from fianco import *
from copy import deepcopy


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
                to = np.stack(np.nonzero(lnd), axis=0, dtype=INT).swapaxes(0, 1)
                fr = to - 2 * np.stack((dy, dx), axis=0, dtype=INT)
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
                to = np.stack(np.nonzero(lnd), axis=0, dtype=INT).swapaxes(0, 1)
                fr = to - np.stack((dy, dx), axis=0, dtype=INT)
                for n in range(to.shape[0]):
                    yield fr[n], to[n]


class BaseAI:
    nc: int

    @classmethod
    def terminal(cls, lst: Engine) -> bool:
        return lst.winner is not None

    @classmethod
    def evaluate(cls, wrt: int, lst: Engine) -> float:
        if lst.winner is not None:
            return (lst.winner == wrt) - (lst.winner != wrt)
        return (lst.brd[wrt].sum() - lst.brd[1 - wrt].sum()) / 15.

    @classmethod
    def αβ_search(cls, wrt: int, lst: Engine, dpth: int, αβ: tuple[float, float]) -> float:
        cls.nc += 1
        print(f"[LOG] searching at depth {dpth} with {αβ} - player {lst.ply}'s turn")
        if dpth <= 0 or cls.terminal(lst):
            return cls.evaluate(wrt, lst)
        α, β = αβ
        scr = -inf
        for frto in chain(capts(lst.ply, lst.brd), steps(lst.ply, lst.brd)):
            assert lst.play(*frto), f'move unsuccessful - tried {frto}'
            val = -cls.αβ_search(wrt, lst, dpth - 1, (-β, -α))
            lst.undo()
            if val > scr:
                scr = val
            if scr > α:
                α = scr
            if scr >= β:
                break
        return scr
    
    @classmethod
    def think(cls, lst: Engine, dpth: int = 3, αβ: tuple[float, float] = (-inf, +inf)) -> Union[tuple[YX, YX], None]:
        cls.nc = 0
        if cls.terminal(lst):
            return None
        wrt = lst.ply
        lst = deepcopy(lst)
        α, β = αβ
        out: Union[tuple[YX, YX], None] = None
        scr = -inf
        for frto in chain(capts(lst.ply, lst.brd), steps(lst.ply, lst.brd)):
            assert lst.play(*frto), f'move unsuccessful - tried {frto}'
            val = -cls.αβ_search(wrt, lst, dpth - 1, (-β, -α))
            lst.undo()
            if val > scr:
                scr = val
                out = frto
        assert out is not None
        return out
    
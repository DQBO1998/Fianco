from typing import TypeAlias
from numpy.typing import NDArray
from numpy.random import default_rng, Generator
from numba.experimental import jitclass # type: ignore
from numba import uint64, float64, uint8, boolean # type: ignore
from fianco import YX, Mat, number # type: ignore

import numpy as np
import numba as nb # type: ignore

Key: TypeAlias = np.uint64
Move: TypeAlias = YX
Score: TypeAlias = np.float64
Flag: TypeAlias = np.uint8
Height: TypeAlias = np.uint64
Player: TypeAlias = np.bool_
RandomNumbers: TypeAlias = NDArray[np.uint64]


EMPTY = 0
BOUND = 1
FINAL = 2


Table: TypeAlias = tuple[NDArray[Key], NDArray[number], NDArray[Score], NDArray[Flag], NDArray[Height], NDArray[Player]]


def make_tt(size: int, rnd: Generator) -> tuple[RandomNumbers, Table]:
    tt_lck = np.array((size,), dtype=Key)
    tt_mov = np.array((size, 2, 2), dtype=number)
    tt_scr = np.array((size,), dtype=Score)
    tt_flg = np.array((size,), dtype=Flag)
    tt_hgt = np.array((size,), dtype=Height)
    tt_wrt = np.array((size,), dtype=Player)
    r_mat = rnd.integers(0, np.iinfo(Key).max, (2, 9, 9), dtype=Key)
    return r_mat, (tt_lck, tt_mov, tt_scr, tt_flg, tt_hgt, tt_wrt)


@nb.njit # type: ignore
def enc_brd(brd: Mat, r_mat: RandomNumbers) -> Key:
    hsh = Key(0)
    for y in range(brd.shape[0]):
        for x in range(brd.shape[1]):
            if np.any(brd[:, y, x]):
                wrt = 1 if brd[1, y, x] else 0
                hsh ^= r_mat[y, x, wrt]
    return hsh


@nb.njit # type: ignore
def hsh2idx(hsh: int, tt: Table) -> int:
    return hsh % len(tt)
    

if __name__ == '__main__':
    r_mat, tt = make_tt(12, default_rng(42))
    print(len(tt))


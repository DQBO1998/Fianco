from typing import TypeAlias
from numpy.typing import NDArray
from numpy.random import Generator
from numba.experimental import jitclass # type: ignore
from fianco import YX, Mat, number # type: ignore

import numpy as np
import numba as nb # type: ignore


LBOUND = 1
UBOUND = 2
FINAL = 3

# Constants for accessing tuple elements
LOCK = 0
FROM = 1
TO = 2
SCORE = 3
FLAG = 4
HEIGHT = 5
AGE = 6
PLAYER = 7  # New constant for player


Hash: TypeAlias = np.uint32
Score: TypeAlias = np.float64
Flag: TypeAlias = np.uint8
Height: TypeAlias = np.int32
Age: TypeAlias = np.uint32
Player: TypeAlias = np.uint8  # Define Player type
RNums: TypeAlias = NDArray[Hash]


# Define types for the arrays
TableLock: TypeAlias = NDArray[Hash]
TableMoveFrom: TypeAlias = NDArray[np.int8]
TableMoveTo: TypeAlias = NDArray[np.int8]
TableScore: TypeAlias = NDArray[Score]
TableFlag: TypeAlias = NDArray[Flag]
TableHeight: TypeAlias = NDArray[Height]
TableAge: TypeAlias = NDArray[Age]
TablePlayer: TypeAlias = NDArray[Player]  # New type for player array

# Define the Table type as a tuple of arrays
Table: TypeAlias = tuple[TableLock, TableMoveFrom, TableMoveTo, TableScore, TableFlag, TableHeight, TableAge, TablePlayer]


def make_tt(size: int, rnd: Generator) -> tuple[RNums, NDArray[Hash], Table]:
    tt_lock = np.zeros(size, dtype=Hash)
    tt_move_from = np.empty((size, 2), dtype=np.int8)
    tt_move_to = np.empty((size, 2), dtype=np.int8)
    tt_score = np.zeros(size, dtype=Score)
    tt_flag = np.zeros(size, dtype=Flag)
    tt_height = np.full(size, -1, dtype=Height)
    tt_age = np.zeros(size, dtype=Age)
    tt_player = np.zeros(size, dtype=Player)  # New array for player
    
    r_mat = rnd.integers(0, np.iinfo(Hash).max - 1, (2, 9, 9), dtype=Hash)
    r_player = rnd.integers(0, np.iinfo(Hash).max - 1, 2, dtype=Hash)
    return r_player, r_mat, (tt_lock, tt_move_from, tt_move_to, tt_score, tt_flag, tt_height, tt_age, tt_player)


@nb.njit # type: ignore
def encode_board(for_player: int, board_state: Mat, r_player: NDArray[Hash], r_mat: RNums) -> Hash:
    hsh = Hash(0)
    for y in range(board_state.shape[1]):
        for x in range(board_state.shape[2]):
            if np.any(board_state[:, y, x]):
                piece_player = 1 if board_state[1, y, x] else 0
                hsh ^= r_mat[piece_player, y, x]
    hsh ^= r_player[for_player]
    return hsh


@nb.njit # type: ignore
def write_tt(index: int, lock: Hash, move_from: YX, move_to: YX, score: Score, flag: Flag, height: Height, age: Age, player: Player, tt: Table) -> None:
    tt[LOCK][index] = lock
    tt[FROM][index] = move_from
    tt[TO][index] = move_to
    tt[SCORE][index] = score
    tt[FLAG][index] = flag
    tt[HEIGHT][index] = height
    tt[AGE][index] = age
    tt[PLAYER][index] = player  # Store the player who made the move


@nb.njit # type: ignore
def read_tt(index: int, tt: Table) -> tuple[Hash, YX, YX, Score, Flag, Height, Age, Player]:
    return (
        tt[LOCK][index],
        tt[FROM][index],
        tt[TO][index],
        tt[SCORE][index],
        tt[FLAG][index],
        tt[HEIGHT][index],
        tt[AGE][index],
        tt[PLAYER][index]  # Return the player who made the move
    )

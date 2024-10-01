
from collections import deque
from collections.abc import Callable
from typing import Any, TypeAlias
from numpy.random import default_rng
from jinja2 import Template
from numpy.typing import NDArray

import fianco as fnc
import bot
import numpy as np


Player: TypeAlias = Callable[[fnc.Engine], tuple[fnc.YX, fnc.YX]]


def smart(ply: int) -> Player:
    def player(game: fnc.Engine) -> tuple[fnc.YX, fnc.YX]:
        _, _, frto = bot.think(game, ply)
        return frto
    return player


def dumb(seed: int = 42) -> Player:
    rnd = default_rng(seed)
    def player(game: fnc.Engine) -> tuple[fnc.YX, fnc.YX]:
        _, _, frto = bot.monke(game, rnd)
        return frto
    return player


def board(brd: fnc.Mat) -> str:
    brd = 1 * brd[0] + 2 * brd[1]
    rows: deque[str] = deque()
    for row in brd:
        row = row.astype(str)
        row[row == '1'] = ' ◻ '
        row[row == '2'] = ' ◼ '
        row[row == '0'] = ' · '
        row = ''.join(c for c in row)
        rows.append(row)
    return '\n'.join(r for r in rows)


def wrt_as_str(wrt: int) -> str:
    if wrt == 0:
        return 'black'
    if wrt == 1:
        return 'white'
    raise NotImplementedError(f'`wrt` should be 0 or 1 - was {wrt}')


_header = Template('========== {{ player }} ==========')
def header(wrt: int) -> str:
    return _header.render(player=wrt_as_str(wrt))

_move = Template('{{ fr }} -> {{ to }}')
def move(fr: fnc.YX, to: fnc.YX) -> str:
    return _move.render(fr=fr, to=to)


def win_as_str(wins: fnc.End | None) -> str:
    if wins == fnc.End.BLACK_WINS:
        return 'black wins!'
    if wins == fnc.End.WHITE_WINS:
        return 'white wins!'
    if wins == fnc.End.TIE:
        return 'tie!'
    return '???'


def trace(max: int = 50, 
          black: Player = smart(9), white: Player = dumb(32), 
          brd: fnc.Mat | None = None) -> tuple[NDArray[Any], NDArray[Any]] | None:
    agents = {0: black, 1: white}
    game = fnc.Engine(brd=brd) if brd is not None else fnc.Engine()
    for _ in range(max):
        if game.wins is not None:
            break
        game.play(*agents[game.wrt](game))
    if game.hst:
        y = game.wins == fnc.End.WHITE_WINS
        H = np.array([h for h in game.hst])
        X = np.unique(H, axis=0)
        return X, np.array([y for _ in range(X.shape[0])])
    return None


def collect(runs: int = 100, max: int = 50, 
            black: Player = smart(9), white: Player = dumb(32)) -> tuple[NDArray[Any], NDArray[Any]] | None:
    X_deque: deque[NDArray[Any]] = deque()
    y_deque: deque[NDArray[Any]] = deque()
    for _ in range(runs):
        Xy = trace(max, black, white)
        if Xy is not None:
            X, y = Xy
            X_deque.append(X)
            y_deque.append(y)
    if X_deque and y_deque:
        X_arr = np.concat([X for X in X_deque], axis=0)
        y_arr = np.concat([y for y in y_deque], axis=0)
        return (X_arr, y_arr)
    return None

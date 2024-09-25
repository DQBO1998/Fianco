from collections.abc import Callable
from typing import TypeAlias
from auto import think
from time import time
from fianco import *


Player: TypeAlias = Callable[[Engine], tuple[int, tuple[YX, YX] | None]]


def fight(max_turns: int = 1000, ply0: Player = lambda g: think(g, 7), ply1: Player = lambda g: think(g, 7)) -> int | None:
    plys = (ply0, ply1)
    game = Engine()
    for t in range(max_turns):
        print(f'#======== {t} ========#')
        print(1 * game.brd[0] + 2 * game.brd[1])
        if game.winner is not None:
            return game.winner
        t0 = time()
        ndc, frto = plys[game.ply](game)
        assert frto is not None
        t1 = time()
        print(f'{frto[0]} ---> {frto[1]} ({ndc} nds / {t1 - t0} s | {ndc / (t1 - t0)} nds / s)')
        game.play(*frto)
    return None


if __name__ == '__main__':
    fight()

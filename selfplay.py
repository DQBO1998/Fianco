from collections.abc import Callable
from textwrap import dedent
from typing import TypeAlias
from bot import think, monke, scr_at
from time import time
from fianco import *


Player: TypeAlias = Callable[[Engine], tuple[int, tuple[YX, YX] | None]]


def fight(max_turns: int = 1000, black: Player = lambda g: think(g, 7), white: Player = lambda g: think(g, 7)) -> int | None:
    agents = (black, white)
    titles = ('black', 'white')
    game = Engine()
    print(str(1 * game.brd[1] + 2 * game.brd[0]).replace('1', 'W').replace('2', 'B'))
    print(dedent(f"""
    affairs:
        W: {scr_at(0, game.end, game.brd)}
        B: {scr_at(1, game.end, game.brd)}
    """).strip())
    for t in range(max_turns):
        print(dedent(f"""
        ========== {titles[game.ply]} | {game.trn} ==========
        thinking...""").strip())
        t0 = time()
        nc, vl, frto = agents[game.ply](game)
        t1 = time()
        fr_yx = frto[0]
        to_yx = frto[1]
        print(dedent(f"""
        choice:
            {int(fr_yx[0]), int(fr_yx[1])} -> {int(to_yx[0]), int(to_yx[1])}
            with score {vl}
        stats:
            Δ = {t1 - t0}
            N = {nc}
            N/Δ = {nc / (t1 - t0 + (ϵ := 1e-42))}
        """).strip())
        game.play(*frto)
        print(str(1 * game.brd[1] + 2 * game.brd[0]).replace('1', 'W').replace('2', 'B'))
        print(dedent(f"""
        affairs:
            W: {scr_at(1, game.end, game.brd)}
            B: {scr_at(0, game.end, game.brd)}
        """).strip())
        if game.winner is not None:
            print(f'winner: {titles[game.winner]}')
            return game.winner
    return None


if __name__ == '__main__':
    winner = fight(black=monke)
    print(f'winner: {winner}')


from collections.abc import Iterator
from itertools import chain, product, cycle, repeat
from dataclasses import dataclass, field
from collections import deque
from textwrap import dedent
from time import time
from typing import TypeAlias
from pydantic import BaseModel
from jinja2 import Template
from getch import pause # type: ignore
from typing_extensions import Self

import pickle as pkl
import json
import numpy as np
import pygame as pyg
import fianco as fnc
import bot3 as bot


Suggestion: TypeAlias = tuple[float, float, bot.Stats, tuple[fnc.YX, fnc.YX]]


def yx2chss(y: int, x: int) -> str:
    if 0 <= y < 9 and 0 <= x < 9:
        col = chr(ord('a') + x)
        row = 8 - y + 1
        return f'{col}{row} {y, x}'
    return '--'


class Settings(BaseModel):
   ply: int
   tts: int
   mxt: int


def load_stng() -> Settings:
    with open('game.json', 'r') as file:
        return Settings.model_validate(json.load(file))


header = Template('========== {{ player }} ==========')
action = Template('action: {{ action }}')
result = Template('result: {{ result }}')
stats = Template(dedent("""ply:     {{ ply }} (ply)
Δt:     {{ delta }} (seconds)
nc:     {{ nodes }} (nodes)
nc/Δt:  {{ nodes / (delta + 1e-15) }} (nodes / seconds)
hits:   {{ hits }} (TT hits)
writes: {{ writes }} (TT writes)
age:    {{ age }} (iteration)
depth:  {{ depth }} (max. depth)
vl:     {{ value }}"""))
move = Template('from {{ fr }} to {{ to }}')


@dataclass
class Game:
    state: fnc.Engine = field(default_factory=lambda: fnc.Engine())
    run: bool = field(default_factory=lambda: True)
    frto: deque[fnc.YX] = field(default_factory=deque)
    disp: pyg.Surface = field(default_factory=lambda: pyg.display.set_mode(((wth := 640), wth)))
    cell: tuple[int, int] = field(default_factory=lambda: (0, 0))
    auto: Suggestion | None = None
    anim: Iterator[str] = field(default_factory=lambda: iter(cycle(chain(*(repeat(c, 7) for c in ['◜', 
                                                                                                  '◠', 
                                                                                                  '◝', 
                                                                                                  '◞', 
                                                                                                  '◡', 
                                                                                                  '◟'])))))
    ply: int = field(default_factory=lambda: load_stng().ply)
    tts: int = field(default_factory=lambda: load_stng().tts)
    mxt: int = field(default_factory=lambda: load_stng().mxt)

    def to_disk(self) -> Self:
        with open('game.pkl', 'wb') as file:
            pkl.dump((self.state.wrt, self.state.brd, self.state.hst), file)
        return self

    def from_disk(self) -> Self:
        try:
            with open('game.pkl', 'rb') as file:
                self.state.wrt, self.state.brd, self.state.hst = pkl.load(file)
            return self
        except FileNotFoundError:
            return self


def get_disp_brd_fac(game: Game) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    disp_wth, disp_hgt = game.disp.get_size()
    brd_hgt, brd_wth = game.state.brd.shape[1:]
    fac_y = int(disp_hgt / brd_hgt)
    fac_x = int(disp_wth / brd_wth)
    return (disp_hgt, disp_wth), (brd_hgt, brd_wth), (fac_y, fac_x)


def get_cursor(fac_y: int, fac_x: int) -> fnc.YX:
    x, y = pyg.mouse.get_pos()
    return np.array((int(y / fac_y), int(x / fac_x)), fnc.number)


def draw_brd(game: Game) -> None:
    game.disp.fill((241, 214, 171))
    _, (brd_hgt, brd_wth), (fac_y, fac_x) = get_disp_brd_fac(game)
    rad = min(fac_y, fac_x) / 3
    for i, j in product(range(brd_hgt), range(brd_wth)):
        xy = (fac_x * (j + 0.5), fac_y * (i + 0.5))
        pyg.draw.circle(game.disp, (0, 0, 0), xy, 3)
        if np.any(game.state.brd[:, i, j] != 0):
            pyg.draw.circle(game.disp, (0, 0, 0), xy, rad)
            color = (50,) * 3
            if game.state.brd[1, i, j] == 1:
                color = (230,) * 3
            pyg.draw.circle(game.disp, color, xy, rad - 0.1 * rad)


def draw_line(game: Game) -> None:
    if len(game.frto) == 1:
        _, _, (fac_y, fac_x) = get_disp_brd_fac(game)
        fr_yx = game.frto[0]
        fr_y, fr_x = fr_yx.astype(np.int16)
        to_y, to_x = get_cursor(fac_y, fac_x).astype(np.int16)
        game.cell = (to_y, to_x)
        pyg.draw.line(game.disp, 
                      (0, 0, 0), 
                      (fr_x * fac_x + fac_x // 2, fr_y * fac_y + fac_y // 2), 
                      (to_x * fac_x + fac_x // 2, to_y * fac_y + fac_y // 2), 
                      1)


def vrts_as_str(game: Game) -> str:
    _, _, (fac_y, fac_x) = get_disp_brd_fac(game)
    y, x = get_cursor(fac_y, fac_x).tolist()
    if len(game.frto) == 0:
        return f'{yx2chss(y, x)}'
    if len(game.frto) == 1:
        fr_y, fr_x = game.frto[0]
        return f'{yx2chss(int(fr_y), int(fr_x))} --> {yx2chss(y, x)}'
    if len(game.frto) == 2:
        fr_y, fr_x = game.frto[0]
        to_y, to_x = game.frto[1]
        return f'{yx2chss(int(fr_y), int(fr_x))} --> {yx2chss(int(to_y), int(to_x))}'
    raise NotImplementedError(f'expected 0 <= len(vrts) <= 2 - got {len(game.frto)}')


def wrt_as_str(wrt: int) -> str:
    if wrt == 0:
        return 'black'
    if wrt == 1:
        return 'white'
    raise NotImplementedError(f'`wrt` should be 0 or 1 - was {wrt}')


def update_title(game: Game) -> None:
    caption = ''
    if game.state.wins == fnc.End.BLACK_WINS:
        caption = 'Black wins!'
    elif game.state.wins == fnc.End.WHITE_WINS:
        caption = 'White wins!'
    elif game.state.wins == fnc.End.TIE:
        caption = 'Tie!'
    elif game.auto is None:
        caption = f'[{wrt_as_str(game.state.wrt)}]: {vrts_as_str(game)}'
    else:
        caption = f'[{wrt_as_str(game.state.wrt)}]: {next(game.anim)}'
    pyg.display.set_caption(caption)


def draw_game(game: Game) -> None:
    draw_brd(game)
    draw_line(game)
    update_title(game)
    pyg.display.flip()


def suggest(game: Game) -> Suggestion:
    t0 = time()
    #vl, frto, stats = bot.iterative_deepening(game.state, game.mxt, game.ply)
    vl, frto, stats = bot.simple(game.state, game.ply)
    t1 = time()
    return (t1 - t0), vl, stats, frto


def main():
    pyg.init()
    try:
        game = Game().from_disk()
        print(f'bots will search {game.ply}-ply')
        print(f'transposition tables size: {game.tts}')
        bot.reset(game.tts)
        print(header.render(player=wrt_as_str(game.state.wrt)))
        clock = pyg.time.Clock()
        while game.run:
            assert 0 <= len(game.frto) <= 2, f'yo, why are you moving {len(game.frto)} steps in one turn?!'
            draw_game(game)
            clock.tick(60)
            for ev in pyg.event.get():
                kmods = pyg.key.get_mods()
                if ev.type == pyg.QUIT:
                    game.run = False
                elif ev.type == pyg.KEYDOWN and ev.key == pyg.K_r and kmods & pyg.KMOD_CTRL:
                    print(action.render(action='reset'))
                    #bot.reset()
                    game.auto = None
                    game = Game()
                    print(header.render(player=wrt_as_str(game.state.wrt)))
                elif ev.type == pyg.KEYDOWN and ev.key == pyg.K_z and kmods & pyg.KMOD_CTRL:
                    ok = game.state.undo()
                    if ok:
                        print(action.render(action='undo'))
                        game.auto = None
                elif ev.type == pyg.KEYDOWN and ev.key == pyg.K_x and kmods & pyg.KMOD_CTRL:
                    if game.auto is None:
                        game.auto = suggest(game)
                elif ev.type == pyg.MOUSEBUTTONDOWN:
                    if ev.button == pyg.BUTTON_RIGHT:
                        game.frto.clear()
                    if ev.button == pyg.BUTTON_LEFT:
                        _, _, (fac_y, fac_x) = get_disp_brd_fac(game)
                        cursor = get_cursor(fac_y, fac_x)
                        game.frto.append(cursor)
            if game.auto is not None:
                Δt, vl, nchits, frto = game.auto
                game.frto.extend(frto)
                print(stats.render(delta=Δt, value=vl, nodes=nchits.nodes, hits=nchits.hits, ply=game.ply, age=nchits.age, writes=nchits.writes, depth=nchits.depth))
                game.auto = None
            while len(game.frto) > 2:
                game.frto.pop()
            if len(game.frto) == 2 and game.auto is None:
                ok = game.state.play(*game.frto)
                if ok:
                    game.to_disk()
                    print(action.render(action='move'))
                    print(move.render(fr=game.frto[0], to=game.frto[1]))
                    print(header.render(player=wrt_as_str(game.state.wrt)))
                game.frto.clear()
    finally:
        pyg.quit()


if __name__ == '__main__':
    main()

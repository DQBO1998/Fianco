
from itertools import product
from dataclasses import dataclass, field
from collections import deque
from fianco import *

import numpy as np
import pygame as pyg


@dataclass
class GameState:
    run: bool = field(default_factory=lambda: True)
    ply: int = field(default_factory=lambda: 1)
    brd: Mat = field(default_factory=lambda: load(r'D:\Github\Fianco\blue.png'))
    msk: Mat = field(default_factory=lambda: load(r'D:\Github\Fianco\goal.png'))
    vrts: deque[YX] = field(default_factory=deque)
    hist: deque[tuple[int, Mat]] = field(default_factory=deque)
    disp: pyg.Surface = field(default_factory=lambda: pyg.display.set_mode(((wth := 640), wth)))


def ply2txt(ply: int) -> str:
    return 'white' if ply else 'black'


def mod2txt(*vrts: YX) -> str:
    substr = ''
    if len(vrts) >= 1:
        substr = f'from {vrts[0] + 1}'
    if len(vrts) >= 2:
        substr = f"{substr} to {vrts[1] + 1}"
    return substr


def paint(cur_yx: YX, gmst: GameState) -> tuple[int, int]:
    gmst.disp.fill((241, 214, 171))
    wth, hgt = gmst.disp.get_size()
    dmi, dmj = gmst.brd.shape[1:]
    cly = int(hgt / dmi)
    clx = int(wth / dmj)
    # draw board
    mat = gmst.brd[0] + 2 * gmst.brd[1]
    rad = min(clx, cly) / 3
    for i, j in product(range(dmi), range(dmj)):
        xy = (j * clx + clx / 2, i * cly + cly / 2)
        pyg.draw.circle(gmst.disp, (0, 0, 0), xy, 3)
        if mat[i, j] != 0:
            pyg.draw.circle(gmst.disp, (0, 0, 0), xy, rad)
            pyg.draw.circle(gmst.disp, (50, 50, 50) if mat[i, j] == 1 else (230, 230, 230), xy, rad - 0.1 * rad)
    # draw line
    caption = ''
    if len(gmst.vrts) == 1:
        fr_yx = gmst.vrts[0]
        fr_y, fr_x = fr_yx
        to_y, to_x = cur_yx
        pyg.draw.line(gmst.disp, (0, 0, 0), (fr_x * clx + clx // 2, fr_y * cly + cly // 2), (to_x * clx + clx // 2, to_y * cly + cly // 2), 1)
        caption = f'{ply2txt(gmst.ply)} | {mod2txt(fr_yx, cur_yx)}'
    else:
        caption = f'{ply2txt(gmst.ply)} | {mod2txt(cur_yx)}'
    pyg.display.set_caption(caption)
    pyg.display.flip()
    return cly, clx


def cursor_at(cly: int, clx: int) -> YX:
    x, y = pyg.mouse.get_pos()
    return np.array((int(y / cly), int(x / clx)), np.int8)


def main():
    pyg.init()
    try:
        gmst = GameState()
        clock = pyg.time.Clock()
        cur_yx = np.array((0, 0), np.int8)
        while gmst.run:
            assert 0 <= len(gmst.vrts) <= 2, f'yo, why are you moving {len(gmst.vrts)} steps in one turn?!'
            cur_yx = cursor_at(*paint(cur_yx, gmst))
            clock.tick(60)
            for ev in pyg.event.get():
                if ev.type == pyg.QUIT:
                    gmst.run = False
                elif ev.type == pyg.KEYDOWN:
                    if ev.key == pyg.K_r:
                        gmst = GameState()
                    elif ev.key == pyg.K_z:
                        kmods = pyg.key.get_mods()
                        if kmods & pyg.KMOD_CTRL:
                            if gmst.hist:
                                gmst.ply, gmst.brd = gmst.hist.pop()
                                gmst.vrts.clear()
                    elif ev.key == pyg.K_x:
                        kmods = pyg.key.get_mods()
                        if kmods & pyg.KMOD_CTRL:
                            # call AI here!
                            # append to gmst.vrts
                            pass
                elif ev.type == pyg.MOUSEBUTTONDOWN:
                    if ev.button == pyg.BUTTON_RIGHT:
                        gmst.vrts.clear()
                    elif ev.button == pyg.BUTTON_LEFT:
                        cur_y, cur_x = cur_yx
                        if gmst.brd[gmst.ply, cur_y, cur_x] == True or (len(gmst.vrts) == 1 and gmst.brd[~gmst.ply, cur_y, cur_x] == False):
                            gmst.vrts.append(cur_yx)
            cnd = all(not win(i, gmst.msk, gmst.brd) for i in (0, 1))
            if len(gmst.vrts) == 2 and cnd:
                fr_yx, to_yx = gmst.vrts
                gmst.vrts.clear()
                old = gmst.brd
                ok, gmst.brd = play(gmst.ply, fr_yx, to_yx, gmst.brd)
                if ok:
                    gmst.hist.append((gmst.ply, old))
                    gmst.ply = 1 - gmst.ply
            if not cnd:
                gmst.vrts.clear()
    finally:
        pyg.quit()

main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : projet_num.py
# Created by Anthony Giraudo the 29/01/2020

"""
"""

# Modules

import numpy as np
import pygame as pg
from pygame.locals import *


# Functions

def initialise_particles(screen, N):
    size_x, size_y = screen.get_size()
    coord_x = np.random.randint(0, size_x, N)
    coord_y = np.random.randint(0, size_y, N)
    for pos in zip(coord_x, coord_y):
        print(pos)
        pg.draw.circle(screen, (0, 0, 255), pos, 10)

# Main
if __name__ == "__main__":
    pg.init()
    display = (1680, 1050)  # taille de la fenetre
    screen = pg.display.set_mode(display)
    screen.fill((255, 255, 255))

    N = 100
    initialise_particles(screen, N)

    pg.display.flip()

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:  # quitter si fenetre fermee
                pg.quit()
                quit()

        pg.display.flip()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : DEM.py
# Created by Anthony Giraudo and Clement Sebastiao the 19/02/2020

"""
"""

# Modules

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import sys


# Functions

def g():
    return 9.81


class Grain:
    __n = 0

    def __init__(self, pos, radius, mass):
        self.radius = radius
        self.pos = np.array(pos, dtype=float)
        self.mass = mass
        self.vel = np.zeros((2,), dtype=float)
        self.acc = np.zeros((2,), dtype=float)
        self.force = np.zeros((2,), dtype=float)
        self.id = Grain.__n
        Grain.__n += 1


class Boite:
    def __init__(self, size, Nx, Ny):
        self.size = size
        self.Nx = Nx
        self.Ny = Ny
        self.cell_size = (size[0]/Nx, size[1]/Ny)
        self.grains = []
        self.grille = np.zeros((self.Nx, self.Ny), dtype=object)
        self.init_grille()

    def init_grille(self):
        self.grille = np.zeros(self.grille.shape, dtype=object)
        for x in range(self.Nx):
            for y in range(self.Ny):
                self.grille[x, y] = []

    def add_grain(self, grain):
        if grain.pos[0] + grain.radius > self.size[0] or grain.pos[0] - grain.radius < 0:
            raise ValueError("Invalid x position for grain : {:.2f}".format(grain.pos[0]))
        if grain.pos[1] + grain.radius > self.size[1] or grain.pos[1] - grain.radius < 0:
            raise ValueError("Invalid y position for grain : {:.2f}".format(grain.pos[1]))
        self.grains += [grain]

    def update_grille(self):
        self.init_grille()
        for grain in self.grains:
            cell_x, cell_y = int(grain.pos[0] // self.cell_size[0]), int(grain.pos[1] // self.cell_size[1])
            self.grille[cell_x, cell_y] += [grain]

    def apply_gravity(self):
        # WARNING: reset all forces
        for grain in self.grains:
            grain.force = np.array([0., -g() * grain.mass])

    def contact(self):
        for x in range(self.Nx):
            for y in range(self.Ny):
                cells = [(x, y), (x-1, y) if x-1>=0 else (), (x-1, y-1) if x-1>=0 and y-1>=0 else (), (x, y-1) if y-1>=0 else (), (x+1, y-1) if x+1 < self.Nx and y-1>=0 else ()]
                print("CURRENT CELL", x, y)
                for grain1 in self.grille[x, y]:
                    print("CURRENT GRAIN", grain1.id)
                    for cell_id, cell in enumerate(cells):
                        if cell: print("cell", cell[0], cell[1])
                        if cell_id == 0:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                print("grain trouve", grain2.id)
                                if grain1.id > grain2.id:
                                    print("contact", grain1.id, grain2.id)
                                    apply_force(grain1, grain2)
                        elif cell:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                print("grain trouve", grain2.id)
                                print("contact", grain1.id, grain2.id)
                                apply_force(grain1, grain2)

    def movement(self, dt):
        # apply velocity verlet scheme
        for grain in self.grains:
            a = grain.force / grain.mass
            grain.vel += (grain.acc + a) * (dt / 2.)
            grain.pos += grain.vel * dt + a * (dt ** 2.) / 2.
            grain.acc = a

        # apply boundary condition
        for grain in self.grains:

            if grain.pos[0] - grain.radius < 0:
                grain.pos[0] = grain.radius
                if grain.vel[0] < 0.:
                    grain.vel[0] *= -.9

            elif grain.pos[0] + grain.radius > self.size[0]:
                grain.pos[0] = self.size[0] - grain.radius
                if grain.vel[0] > 0.:
                    grain.vel[0] *= -.9

            if grain.pos[1] - grain.radius < 0:
                grain.pos[1] = grain.radius
                if grain.vel[1] < 0.:
                    grain.vel[1] *= -.9

            elif grain.pos[1] + grain.radius > self.size[1]:
                grain.pos[1] = self.size[1] - grain.radius
                if grain.vel[1] > 0.:
                    grain.vel[1] *= -.9

    def loop_function(self, dt):
        self.update_grille()
        self.apply_gravity()
        self.contact()
        self.movement(dt)


def apply_force(grain1, grain2):
    if grain1.id == grain2.id:
        raise ValueError("Objects grain1 and grain2 have the same id, should not compute contact forces")
    stiffness = 10000.
    damping = 14.
    dist = np.linalg.norm(grain2.pos - grain1.pos)
    delta = -dist + grain1.radius + grain2.radius
    normal = (grain2.pos - grain1.pos) / dist
    if delta > 0.:
        f = normal * delta * stiffness
        grain1.force -= f
        grain2.force += f

        # manage damping factor
        diff_vel = np.dot((grain2.vel - grain1.vel), normal)
        F = damping * diff_vel * normal
        grain1.force += F
        grain2.force -= F


def animate(i, ax, boite, dt, n_skip_drawing, max_iteration):
    print('computing iteration', i*n_skip_drawing, '/', max_iteration)

    ########debug
    for grain in boite.grains:
        print("vel", grain.vel)
    ###########

    for k in range(n_skip_drawing + 1):
        boite.loop_function(dt)

    iteration = i * n_skip_drawing # TODO: a arranger, c un peu le bordel
    if iteration >= max_iteration:
        sys.exit(0)

    # TODO: a capter !!
    for grain in boite.grains:
        if hasattr(grain, 'patch') is False:
            grain.patch = plt.Circle((grain.pos[0], grain.pos[1]), grain.radius)
            ax.add_patch(grain.patch)
        grain.patch.center = (grain.pos[0], grain.pos[1])

    patch_list = [grain.patch for grain in boite.grains]

    ########## debug
    for x in range(boite.size[0]//boite.Nx, boite.size[0], boite.size[0]//boite.Nx):
        plt.plot([x, x], [0, boite.size[1]])
    for y in range(boite.size[1]//boite.Ny, boite.size[1], boite.size[1]//boite.Ny):
        plt.plot([0, boite.size[0]], [y, y])

    plt.xlabel("iteration={}".format(iteration))
    """
    a = np.zeros(boite.grille.shape, dtype=object)
    for x in range(boite.Nx):
        for y in range(boite.Ny):
            a[x, y] = []
            for grain in boite.grille[x, y]:
                a[x, y] += [grain.id]
    print(np.flip(a.T, axis=0))
    """
    ################

    return patch_list


# Main

if __name__ == "__main__":
    max_iteration = 200
    n_skip_drawing = 1 # TODO: en vrai n_skip_drawing skip n-1 frame, pas clair à arranger

    size = (100, 100)
    N = 3

    r = np.amin(size)//N/2  # TODO: faudra faire gaffe aux unites
    m = 1  # TODO: faudra faire gaffe aux unites

    dt = 0.006

    Nx = int(np.around(size[0] / (r*2 * 1.1)))  # TODO: a modifier eventuellement
    Ny = int(np.around(size[1] / (r*2 * 1.1)))  # TODO: a modifier eventuellement
    print("boites", Nx, Ny)

    boite = Boite(size, Nx, Ny)

    print(size[0]//N, r)
    for x in range(size[0]//N, size[0] - size[0]//N, size[0]//N):
        for y in range(size[1]//N, size[1] - size[1]//N, size[1]//N):
            x_rand = x + np.random.normal()
            if x == size[0]//N:
                x_rand += 10
            y_rand = y + np.random.normal()
            print(x, y)
            boite.add_grain(Grain([x_rand, y_rand], r, m))

    # init matplotlib figure
    fig = plt.figure()
    plt.xlim(0, size[0])
    plt.ylim(0, size[1])
    ax = plt.axes()  # TODO: a capter
    plt.gca().set_aspect('equal', adjustable='box')  # TODO: a capter

    anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=max_iteration, interval=50, fargs=(ax, boite, dt, n_skip_drawing, max_iteration), repeat=False)
    #plt.show()
    anim.save('test.mp4', metadata={'artist': 'Guido'})

# TODO : qu'on puisse voir la premiere frame !
# TODO : checker que les contacts se font qu'une fois a chaque fois entre 2 particules, je suis pas sur sur que c au point
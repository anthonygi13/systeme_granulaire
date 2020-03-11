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
        self.pos = np.array(pos)
        self.mass = mass
        self.vel = np.zeros((2,))
        self.acc = np.zeros((2,))
        self.force = np.zeros((2,))
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
        #TODO: checker aussi superposition avec d'autres grains ?
        if grain.pos[0] + grain.radius > self.size[0] or grain.pos[0] - grain.radius < 0:
            raise ValueError("Invalid x position for grain : {:.2f}".format(grain.pos[0]))
        if grain.pos[1] + grain.radius > self.size[1] or grain.pos[1] - grain.radius < 0:
            raise ValueError("Invalid y position for grain : {:.2f}".format(grain.pos[1]))
        self.grains += [grain]  # TODO: utile sachant qu'on a deja tous les grains dans la grille ? pour l instant utile dans update_grille mais on peut le virer
        cell_x, cell_y = int(grain.pos[0] / self.cell_size[0]), int(grain.pos[1] / self.cell_size[1])
        """
        print("GRAIIIIIIIIIINNNNNNNN")
        print("pos grain", grain.pos)
        print("coord cell", cell_x, cell_y)
        print("cell size", self.cell_size)
        print("Nx", self.Nx)
        print("Ny", self.Ny)
        print("pos cell", self.cell_size[0]*cell_x, self.cell_size[1]*cell_y)
        """
        self.grille[cell_x, cell_y] += [grain]

    def update_grille(self):
        self.init_grille()
        for grain in self.grains:
            cell_x, cell_y = int(grain.pos[0] // self.cell_size[0]), int(grain.pos[1] // self.cell_size[1])
            self.grille[cell_x, cell_y] += [grain]

    def contact(self):
        for x in range(self.Nx):
            for y in range(self.Ny):
                cells = [(x, y), (x-1, y) if x-1>0 else (), (x-1, y-1) if x-1>0 and y-1>0 else (), (x, y-1) if y-1>0 else (), (x+1, y-1) if x+1 < self.Nx and y-1>0 else ()]
                for grain1 in self.grille[x, y]:
                    for cell_id, cell in enumerate(cells):
                        if cell_id == 0:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                if grain1.id > grain2.id:
                                    apply_force(grain1, grain2)
                        elif cell:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                apply_force(grain1, grain2)

    def apply_gravity(self):
        # WARNING: reset all forces
        for grain in self.grains:
            grain.force = np.array([0., -g() * grain.mass])

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
        self.apply_gravity()
        self.contact()
        self.movement(dt)
        self.update_grille()


def apply_force(grain1, grain2):
    if grain1.id == grain2.id:
        raise ValueError("Objects grain1 and grain2 have the same id, should not compute contact forces")
    stiffness = 10000.
    damping = 14.
    dist = np.linalg.norm(grain2.pos - grain1.pos)
    delta = -dist + grain1.radius + grain2.radius
    normal = (grain2.pos - grain1.pos) / dist
    if (delta > 0.):
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

    for k in range(n_skip_drawing):
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
    """"
    ########## debug
    a = np.zeros(boite.grille.shape, dtype=object)
    for x in range(boite.Nx):
        for y in range(boite.Ny):
            a[x, y] = []
            for grain in boite.grille[x, y]:
                a[x, y] += [grain.id]
    print(a)

    ################
    """
    return patch_list


# Main

if __name__ == "__main__":
    max_iteration = 1000
    n_skip_drawing = 1

    r = 4  # TODO: faudra faire gaffe aux unites
    m = 1  # TODO: faudra faire gaffe aux unites

    dt = 0.006

    size = (100, 100)
    Nx = int(np.around(size[0] / (r * 1.1)))  # TODO: a modifier eventuellement
    Ny = int(np.around(size[1] / (r * 1.1)))  # TODO: a modifier eventuellement

    boite = Boite(size, Nx, Ny)

    for x in range(10, 100, 10):
        for y in range(10, 100, 10):
            x_rand = x + np.random.normal()
            y_rand = y + np.random.normal()
            boite.add_grain(Grain([x_rand, y_rand], r, m))

    # init matplotlib figure
    fig = plt.figure()
    plt.xlim(0, size[0])
    plt.ylim(0, size[1])
    ax = plt.axes()
    plt.gca().set_aspect('equal', adjustable='box')  # TODO: a capter

    anim = animation.FuncAnimation(fig, animate, frames=max_iteration, interval=50, fargs=(ax, boite, dt, n_skip_drawing, max_iteration), repeat=False)
    #plt.show()
    anim.save('test.mp4', metadata={'artist': 'Guido'})

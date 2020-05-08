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
    return 0.981 * 1000  # cm par s^2


class Grain:
    __n = 0

    def __init__(self, pos, radius, mass):
        self.radius = radius
        self.pos = np.array(pos, dtype=float)
        self.mass = mass
        self.vel = np.zeros((2,), dtype=float)
        self.acc = np.zeros((2,), dtype=float)
        self.force = np.zeros((2,), dtype=float)
        self.cell_x = -1
        self.cell_y = -1
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
            grain.cell_x = cell_x
            grain.cell_y = cell_y

    def apply_gravity(self):
        # WARNING: reset all forces
        for grain in self.grains:
            grain.force = np.array([0., -g() * grain.mass])

    def contact(self):
        visited_cells = []
        for grain in self.grains:
            x, y = grain.cell_x, grain.cell_y
            if not (x, y) in visited_cells:
                cells = [(x, y), (x-1, y) if x-1>=0 else (), (x-1, y-1) if x-1>=0 and y-1>=0 else (), (x, y-1) if y-1>=0 else (), (x+1, y-1) if x+1 < self.Nx and y-1>=0 else ()]
                for grain1 in self.grille[x, y]:
                    for cell_id, cell in enumerate(cells):
                        if cell_id == 0:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                if grain1.id > grain2.id:
                                    apply_force(grain1, grain2)
                        elif cell:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                apply_force(grain1, grain2)

    def normal_bc(self):
        restitution_coeff = 0.3
        for grain in self.grains:

            if grain.pos[0] - grain.radius < 0:
                grain.pos[0] = grain.radius
                if grain.vel[0] < 0.:
                    grain.vel[0] *= -restitution_coeff

            elif grain.pos[0] + grain.radius > self.size[0]:
                grain.pos[0] = self.size[0] - grain.radius
                if grain.vel[0] > 0.:
                    grain.vel[0] *= -restitution_coeff

            if grain.pos[1] - grain.radius < 0:
                grain.pos[1] = grain.radius
                if grain.vel[1] < 0.:
                    grain.vel[1] *= -restitution_coeff

            elif grain.pos[1] + grain.radius > self.size[1]:
                grain.pos[1] = self.size[1] - grain.radius
                if grain.vel[1] > 0.:
                    grain.vel[1] *= -restitution_coeff

    def sablier_bc(self):
        self.normal_bc()
        pass  # TODO

    def movement(self, dt, boundary_conditions):

        # apply velocity verlet scheme
        for grain in self.grains:
            a = grain.force / grain.mass
            grain.vel += (grain.acc + a) * (dt / 2.)
            grain.pos += grain.vel * dt + a * (dt ** 2.) / 2.
            grain.acc = a

        # boundary conditions
        boundary_conditions(self)

    def loop_function(self, dt, boundary_conditions):
        self.update_grille()
        self.apply_gravity()
        self.contact()
        self.movement(dt, boundary_conditions)


def apply_force(grain1, grain2):
    if grain1.id == grain2.id:
        raise ValueError("Objects grain1 and grain2 have the same id, should not compute contact forces.")
    dist = np.linalg.norm(grain2.pos - grain1.pos)
    delta = -dist + grain1.radius + grain2.radius

    if dist == 0.:
        raise ValueError("Same position for grain1 and grain2, maybe the time step should be reduced.")

    normal = (grain2.pos - grain1.pos) / dist

    if delta > 0.:
        """
        # hertz model for contact
        E = 80e10  # young modulus, g.cm^-1.s^-2
        mu = 0.22  # poisson ratio
        f = 2*E/(3 * (1-mu**2)) * np.sqrt(1/(1/grain1.radius + 1/grain2.radius)) * delta**(3/2) * normal
        grain1.force -= f
        grain2.force += f
        """

        # linear spring dashpot contact model
        stiffness = 10000. * 1000  # g par seconde^2
        damping = 14. * np.sqrt(1000)  # g par seconde
        f = normal * delta * stiffness
        grain1.force -= f
        grain2.force += f
        # manage damping factor
        diff_vel = np.dot((grain2.vel - grain1.vel), normal)
        F = damping * diff_vel * normal
        grain1.force += F
        grain2.force -= F


def animate(i, ax, boite, dt, n_skip_drawing, max_iteration, boundary_conditions):
    print('computing iteration', i*(n_skip_drawing+1), '/', max_iteration)

    if i != 0:
        for k in range(n_skip_drawing + 1):
            boite.loop_function(dt, boundary_conditions)

    iteration = i * (n_skip_drawing + 1)
    if iteration >= max_iteration:
        sys.exit(0)

    for grain in boite.grains:
        if hasattr(grain, 'patch') is False:
            grain.patch = plt.Circle((grain.pos[0], grain.pos[1]), grain.radius)
            ax.add_patch(grain.patch)
        grain.patch.center = (grain.pos[0], grain.pos[1])

    patch_list = [grain.patch for grain in boite.grains]

    plt.xlabel("iteration={}".format(iteration))

    return patch_list


# Main

if __name__ == "__main__":
    boundary_conditions = Boite.normal_bc
    max_iteration = 20000
    n_skip_drawing = 4
    taille_boite = (20, 20)  # cm
    N = 10
    espacement = 0.03  # cm
    r = 0.4  # cm
    initial_pos = ((taille_boite[0] - N*(2*r+espacement)) / 2, (taille_boite[1] - N*(2*r+espacement)) / 2 - 5)
    density = 2700 * 1000 / 100**3  # g/cm^3
    #m = 4/3 * np.pi * r**3 * density  # g
    m = 10  # g
    dt = 0.006 / np.sqrt(1000)  # s

    Nx = int(np.around(taille_boite[0] / (r * 2 * 1.1)))
    Ny = int(np.around(taille_boite[1] / (r * 2 * 1.1)))
    boite = Boite(taille_boite, Nx, Ny)

    for i in range(N):
        for j in range(N):
            x = initial_pos[0] + i * (2*r + espacement) + np.random.normal() * espacement/4
            y = initial_pos[1] + j * (2*r + espacement) + np.random.normal() * espacement/4
            boite.add_grain(Grain((x, y), r, m))

    # init matplotlib figure
    fig = plt.figure()
    plt.xlim(0, taille_boite[0])
    plt.ylim(0, taille_boite[1])
    ax = plt.axes()
    plt.gca().set_aspect('equal', adjustable='box')

    anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=max_iteration, interval=dt*1000*(n_skip_drawing+1), fargs=(ax, boite, dt, n_skip_drawing, max_iteration, boundary_conditions), repeat=False)
    #plt.show()
    anim.save('test.mp4', metadata={'artist': 'Guido'})

# TODO : sablier, et ajuster valeurs pour que ce soit pas mal, en fait poser equations pour retrouver comportement du code qu'on avait chope

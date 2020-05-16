#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : DEM.py
# Authors : Anthony Giraudo and Clement Sebastiao

""" Methode des elements discrets pour la simulation 2D d'un systeme granulaire
"""

# Modules

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import sys


# Functions

def g():
    """
    :return: acceleration de la pesanteur
    """
    return 981  # cm par s^2


class Grain:
    """Classe pour gerer les grains comme des objets
    """

    __n = 0

    def __init__(self, pos, radius, mass):
        """
        :param pos: tuple (x, y), position du grain
        :param radius: rayon du grain
        :param mass: masse du grain
        """
        self.radius = radius
        self.pos = np.array(pos, dtype=float)
        self.mass = mass
        self.vel = np.zeros((2,), dtype=float)  # vitesse
        self.acc = np.zeros((2,), dtype=float)  # acceleration
        self.force = np.zeros((2,), dtype=float)  # force ressentie
        self.id = Grain.__n  # identifiant unique
        Grain.__n += 1


class Boite:
    """Classe pour gerer l'evolution du systeme granulaire
    """

    def __init__(self, size, Nx, Ny):
        """
        :param size: tuple (size_x, size_y), dimension de la boite
        :param Nx: nombre de colonnes pour optimisation par quadrillage spatial
        :param Ny: nomrbe de lignes pour optimisation par quadrillage spatial
        """
        self.size = size
        self.Nx = Nx
        self.Ny = Ny
        self.cell_size = (size[0]/Nx, size[1]/Ny)  # dimensions d'une case pour le quadrillage spatial
        self.grains = []  # contiendra les objets grains du système
        # tableau correspondant au quadrillage spatial, chaque case contiendra la liste des grains qui s'y situent
        self.grille = np.zeros((self.Nx, self.Ny), dtype=object)
        self.init_grille()  # initialisation de self.grille avec des listes vides

    def init_grille(self):
        """Initialisation de self.grille avec des listes vides
        """
        # pour chaque case
        for x in range(self.Nx):
            for y in range(self.Ny):
                # attribuer une liste vide
                self.grille[x, y] = []

    def add_grain(self, grain):
        """
        Ajout d'un grain dans le systeme
        :param grain: objet grain
        """
        # verification de la position
        if grain.pos[0] + grain.radius > self.size[0] or grain.pos[0] - grain.radius < 0:
            raise ValueError("Invalid x position for grain : {:.2f}".format(grain.pos[0]))
        if grain.pos[1] + grain.radius > self.size[1] or grain.pos[1] - grain.radius < 0:
            raise ValueError("Invalid y position for grain : {:.2f}".format(grain.pos[1]))

        self.grains += [grain]  # ajout du grain dans la liste de grains

    def update_grille(self):
        """Mise a jour de self.grille en fonction des positions des grains
        """
        self.init_grille()  # reinitialiser la grille avec des listes vides
        for grain in self.grains:  # pour chaque grain
            # calculer la case correspondante
            cell_x, cell_y = int(grain.pos[0] // self.cell_size[0]), int(grain.pos[1] // self.cell_size[1])
            # placer le grain dans la case correspondante
            self.grille[cell_x, cell_y] += [grain]

    def apply_gravity(self):
        # BE CAREFUL: reset all forces
        for grain in self.grains:  # pour chaque grain
            # la force ressentie est celle de la pesanteur
            grain.force = np.array([0., -g() * grain.mass])

    def contact(self):
        """Cherche les grains potentiellement en contact
        """
        visited_cells = []  # liste des cases deja traitees
        for grain in self.grains:  # pour chaque grain
            # regarder la case dans laquelle il se situe
            x, y = int(grain.pos[0] // self.cell_size[0]), int(grain.pos[1] // self.cell_size[1])
            if not (x, y) in visited_cells:  # si on a pas deja traite la case
                visited_cells += [(x, y)]  # on marque la case
                # calcul des cases adjacentes a traiter
                cells = [(x, y), (x-1, y) if x-1>=0 else (), (x-1, y-1) if x-1>=0 and y-1>=0 else (), (x, y-1) if y-1>=0 else (), (x+1, y-1) if x+1 < self.Nx and y-1>=0 else ()]
                for grain1 in self.grille[x, y]:  # pour chaque grain dans la case
                    #  pour chaque case a traiter, pour chaque grain dans ces cases appeler la fonction
                    #  qui testera le contact avec le premier grain, et qui appliquera les forces le cas echeant
                    for cell_id, cell in enumerate(cells):
                        if cell_id == 0:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                if grain1.id > grain2.id:
                                    apply_force(grain1, grain2)
                        elif cell:
                            for grain2 in self.grille[cell[0], cell[1]]:
                                apply_force(grain1, grain2)

    def bc(self):
        """Applique conditions aux bords
        """
        restitution_coeff = 0.3
        # pour chaque grain rectifier sa position et sa vitesse si en dehors de la boite
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

    def movement(self, dt):
        """
        Deplace les grains pour une etape de temps
        :param dt: pas de temps utilise
        """

        # apply velocity verlet scheme
        for grain in self.grains:
            a = grain.force / grain.mass
            grain.vel += (grain.acc + a) * (dt / 2.)
            grain.pos += grain.vel * dt + a * (dt ** 2.) / 2.
            grain.acc = a

        # boundary conditions
        self.bc()

    def loop_function(self, dt):
        """
        Etapes a repeter
        :param dt: pas de temps utilise
        """
        self.update_grille()
        self.apply_gravity()
        self.contact()
        self.movement(dt)


def apply_force(grain1, grain2):
    """
    Applique les forces de contact entre deux grains si ils sont en contact
    :param grain1: objet grain
    :param grain2: objet grain
    """
    if grain1.id == grain2.id:
        raise ValueError("Objects grain1 and grain2 have the same id, should not compute contact forces.")

    dist = np.linalg.norm(grain2.pos - grain1.pos)
    delta = -dist + grain1.radius + grain2.radius  # distance d'interpenetration

    if dist == 0.:
        raise ValueError("Same position for grain1 and grain2, maybe the time step should be reduced.")

    normal = (grain2.pos - grain1.pos) / dist  # vecteur normal

    if delta > 0.:  # si contact

        # linear spring dashpot contact model
        # raideur
        stiffness = 10000000.  # g par seconde^2
        # amortissement
        damping = 443.  # g par seconde
        # calcul et application des forces
        f = normal * delta * stiffness
        grain1.force -= f
        grain2.force += f
        # manage damping factor
        diff_vel = np.dot((grain2.vel - grain1.vel), normal)
        F = damping * diff_vel * normal
        grain1.force += F
        grain2.force -= F


def animate(i, ax, boite, dt, n_skip_drawing, max_iteration):
    """
    Permet de creer l'animation matplotlib du systeme simule
    :param i: numero de frame
    :param ax: objet Axes de matplotlib
    :param boite: objet boite correspondant au systeme a simuler
    :param dt: pas de temps
    :param n_skip_drawing: nombre d'etapes a calculer sans afficher a chaque iteration
    :param max_iteration: limite d'iteration
    """

    print('computing iteration', i*(n_skip_drawing+1), '/', max_iteration)

    # calcule un certain nombre d'etaoes
    if i != 0:
        for k in range(n_skip_drawing + 1):
            boite.loop_function(dt)

    # arret si on depasse max_iteration
    iteration = i * (n_skip_drawing + 1)
    if iteration >= max_iteration:
        sys.exit(0)

    # gere l'affichage de la frame
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
    # parametres
    max_iteration = 10000  # limite d'iteration
    n_skip_drawing = 99  # nombre d'etapes a ne pas afficher entre 2 frames
    taille_boite = (20, 20)  # cm
    N = 10  # racine carré du nombre de particule total
    r = 0.4  # rayon des grains, cm
    espacement = r / 10  # espacement initiale entre deux grains, cm
    initial_pos = ((taille_boite[0] - N*(2*r+espacement)) / 2, (taille_boite[1] - N*(2*r+espacement)) * 1/4)
    m = 1  # masse des grains, g
    dt = 0.0002  # s
    Nx = int(np.around(taille_boite[0] / (r * 2 * 1.1)))  # nombre de colonnes pour optimisation par quadrillage spatial
    Ny = int(np.around(taille_boite[1] / (r * 2 * 1.1)))  # nombre de lignes pour optimisation par quadrillage spatial

    # initialisation
    boite = Boite(taille_boite, Nx, Ny)
    # pour chaque grain, le placer, avec un petit decalage aleatoire
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

    # calcul et animation
    anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=max_iteration, interval=dt*1000*(n_skip_drawing+1), fargs=(ax, boite, dt, n_skip_drawing, max_iteration), repeat=False)
    #plt.show()
    anim.save('out.mp4')

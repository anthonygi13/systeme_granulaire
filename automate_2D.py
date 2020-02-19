#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : automate_2D.py
# Created by Anthony Giraudo and Clement Sebastao the 05/02/2020

"""
"""

# Modules

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Functions

def initialisation_automate(n=100, p=0.3):
    return np.random.binomial(1, p, (n, n))


def check_sommet(grille, x, y):
    if x+1 == grille.shape[0] or grille[x+1, y] == 2:
        return True
    elif grille[x+1, y] == 0:
        return False
    elif grille[x+1, y] == 1:
        return check_sommet(grille, x+1, y)


def get_sommets(grille):
    sommets = grille == 1
    sommets[1:, :] *= (grille[:-1, :] == 0)
    indices = np.argwhere(sommets)
    for indice in indices:
            x, y = indice[0], indice[1]
            if not check_sommet(grille, x, y):
                sommets[x, y] = False
    return sommets


def etape(grille):

    # FIXME: une tour qui fait toute la hauteur ne s'ecroule pas

    old_grille = np.array(grille, copy=True)

    masque_chute = np.zeros(old_grille.shape, dtype=bool)
    masque_chute[:-1, :] = old_grille[1:, :] == 0
    masque_chute *= old_grille == 1

    grille[masque_chute] = 0
    grille[1:, :][masque_chute[:-1, :]] = 1

    #sommets = old_grille.shape[0] - np.argmin(np.flip(old_grille, axis=0), axis=0)
    # x = np.indices(old_grille.shape)[0]
    sommets = get_sommets(old_grille)
    pas_voisin_droite = np.zeros(old_grille.shape, dtype=bool)
    pas_voisin_droite[:, :-1] = old_grille[:, 1:] == 0
    pas_voisin_gauche = np.zeros(old_grille.shape, dtype=bool)
    pas_voisin_gauche[:, 1:] = old_grille[:, :-1] == 0
    pas_voisin_bas_droite = np.zeros(old_grille.shape, dtype=bool)
    pas_voisin_bas_droite[:-1, :-1] = old_grille[1:, 1:] == 0
    pas_voisin_bas_gauche = np.zeros(old_grille.shape, dtype=bool)
    pas_voisin_bas_gauche[:-1, 1:] = old_grille[1:, :-1] == 0
    futur_voisin_bas_droite = np.zeros(old_grille.shape, dtype=bool)
    futur_voisin_bas_droite[:-1, :-1] = masque_chute[1:, 1:]
    futur_voisin_bas_gauche = np.zeros(old_grille.shape, dtype=bool)
    futur_voisin_bas_gauche[:-1, 1:] = masque_chute[1:, :-1]

    masque_droite = sommets * pas_voisin_droite * pas_voisin_bas_droite * np.logical_not(futur_voisin_bas_droite)
    masque_gauche = sommets * pas_voisin_gauche * pas_voisin_bas_gauche * np.logical_not(futur_voisin_bas_gauche)
    masque_gauche_pas_droite = masque_gauche * np.logical_not(masque_droite)
    masque_droite_pas_gauche = masque_droite * np.logical_not(masque_gauche)
    masque_droite_gauche = masque_droite * masque_gauche

    grille[masque_droite + masque_gauche] = 0

    rand = np.random.binomial(1, 0.5, old_grille.shape).astype(bool)
    grille[1:, 1:][(masque_droite_gauche * rand)[:-1, :-1]] = 1  # gere les chutes a droite
    grille[1:, :-1][(masque_droite_gauche * np.logical_not(rand))[:-1, 1:]] = 1  # gere les chutes a gauche

    grille[1:, 1:][masque_droite_pas_gauche[:-1, :-1]] = 1
    grille[1:, :-1][masque_gauche_pas_droite[:-1, 1:]] = 1


def animate(i, grille, plot):
    etape(grille)
    plot.set_array(grille)
    return plot,


# Main

if __name__ == "__main__":
    #grille = initialisation_automate()
    grille = np.array([[0, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
    """
    n = 100
    grille = np.zeros((n, n))
    grille[:50, 0] = 1
    for i in range(20):
        grille[50+i, i] = 2
    grille[80, 18:23] = 2
    """
    #grille[50:, 40:60] = np.ones((50, 20))
    fig = plt.figure()
    plot = plt.imshow(grille)
    anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=1000, interval=1000, fargs=(grille, plot), repeat=False)
    plt.show()

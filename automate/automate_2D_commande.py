#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : automate_2D_commande.py
# Authors : Anthony Giraudo and Clement Sebastiao

"""
Ce code permet de creer un automate cellulaire simulant la chute de grains de sable (voir le README.txt pour mode
d'emploi). Notre systeme est represente dans un array de valeur, le vide est represente par 0, les grains par 1,
les surfaces par 2, et les bords agissent comme des surfaces.
"""


# Modules

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
from fonctions_automate_v1 import *
from fonctions_automate_v2 import *


# Main

if __name__ == "__main__":
    # On cree un parser qui permet a l'utilisateur de renseigner des arguments optionnels a l'execution

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", "-v", help="Choose the used version of the cellular automaton", type=int,
                        choices=[1, 2])
    parser.add_argument("--structure", "-s", help="Choose the displayed structure in the cellular automaton",
                        choices=["flow", "f", "avalanche", "a", "hourglass", "h", "galton", "g"])
    args = parser.parse_args()

    # On initialise notre grille, et on part d'une configuration initiale en fonction l'argument entre.
    n = 100
    grille = np.zeros((n, n))
    if args.structure == "flow" or args.structure == "f":
        grille[:49, 0] = 1
        for i in range(20):
            grille[50 + i, i] = 2
            grille[80, 18:23] = 2
    elif args.structure == "avalanche" or args.structure == "a":
        grille[0, 0] = 2  # bricolage pour l'affichage des bonnes couleurs
        pos = 50
        taille = [i for i in range(15, 0, -2)]
        grille[-2:, pos - taille[0] // 2:pos + taille[0] // 2 + 1] = 1
        for i in range(1, len(taille)):
            grille[-i * 2 - 2:-i * 2, pos - taille[i] // 2:pos + taille[i] // 2 + 1] = 1

        grille[-len(taille) * 2 - 10, pos + 4] = 1
    elif args.structure == "hourglass" or args.structure == "h":
        for i in range(19, 80):
            grille[i + 20, i] = 2
            grille[i + 20, 99 - i] = 2
        for k in range(25, 70):
            for j in range(19, 40):
                grille[j - 19, k + 2] = np.random.binomial(1, 0.5)
        grille[69:71, 49:51] = 0
        grille[69:71, 48] = 2
        grille[69:71, 51] = 2
    elif args.structure == "galton" or args.structure == "g":
        grille = np.zeros((160, 100))
        for i in range(25, 50):
            grille[i - 25, i] = 2
            grille[i - 25, 100 - i] = 2
            grille[i - 25, i + 1:100 - i] = 1
        for i in range(0, 20):
            grille[79:, 31 + 2 * i] = 2
        for i in range(0, 19):
            for j in range(0, i):
                grille[25 + 3 * i, 51 - i + 2 * j] = 2
    else:
        print("No structure specified, hourglass displayed by default")
        for i in range(19, 80):
            grille[i + 20, i] = 2
            grille[i + 20, 99 - i] = 2
        for k in range(25, 70):
            for j in range(19, 40):
                grille[j - 19, k + 2] = np.random.binomial(1, 0.5)
        grille[69:71, 49:51] = 0
        grille[69:71, 48] = 2
        grille[69:71, 51] = 2

    fig = plt.figure()
    plot = plt.imshow(grille, 'afmhot')
    plt.axis('off')
    

    #  Suivant l'argument entre on utilise une certaine version, de notre automate cellulaire, puis on anime.
    if args.version == 1:
        anim = animation.FuncAnimation(fig, animate1, init_func=lambda: None, frames=2000, interval=50,
                                       fargs=(grille, plot), repeat=False)
    elif args.version == 2:
        anim = animation.FuncAnimation(fig, animate2, init_func=lambda: None, frames=2000, interval=50,
                                       fargs=(grille, plot), repeat=False)
    else:
        print("No version specified, version 1 used by default")
        anim = animation.FuncAnimation(fig, animate1, init_func=lambda: None, frames=2000, interval=50,
                                       fargs=(grille, plot), repeat=False)
    plt.show()
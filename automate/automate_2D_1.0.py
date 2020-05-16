#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : automate_2D.py
# Created by Anthony Giraudo and Clement Sebastiao the 05/02/2020

"""
Ce code permet de creer un automate cellulaire simulant la chute de grains de sable (voir le README.txt).
Notre systeme est represente dans un array de valeur, le vide est represente par 0, les grains par 1, 
les surfaces par 2, et les bords agissent comme des surfaces.
"""

# Modules

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# Functions

"Fonction recursive permettant de tester si la case verifiee est le sommet d'une colonne de sable ou non."
def check_sommet(grille, i, j):
    """
    :param grille: array, espace de l'automate cellulaire
    :param i: integer, ligne de la case à verifier
    :param j: integer, colonne de la case à verifier
    """
    if i+1 == grille.shape[0] or grille[i+1, j] == 2:
        return True
    elif grille[i+1, j] == 0:
        return False
    elif grille[i+1, j] == 1:
        return check_sommet(grille, i+1, j)

"Fonction permettant de recuperer la liste des sommets des colonnes de sable dans notre grille."
def get_sommets(grille):
    """
    :param grille: array, espace de l'automate cellulaire
    """
    sommets = grille == 1
    sommets[1:, :] *= (grille[:-1, :] == 0)
    indices = np.argwhere(sommets)
    for indice in indices:
            i, j = indice[0], indice[1]
            if not check_sommet(grille, i, j):
                sommets[i, j] = False
    return sommets

"Fonction actualisant notre automate cellulaire d'un pas de temps."
def etape(grille):
    """
    :param grille: array, espace de l'automate cellulaire
    """
    
    
    old_grille = np.array(grille, copy=True)
    
    """
    On commence par faire chuter tous les grains de sable sous lesquelles il n'y a pas de grain.
    """
    masque_chute = np.zeros(old_grille.shape, dtype=bool)
    masque_chute[:-1, :] = old_grille[1:, :] == 0
    masque_chute *= old_grille == 1

    grille[masque_chute] = 0
    grille[1:, :][masque_chute[:-1, :]] = 1
    
    """
    On met en place les differents masques nous renseignant sur la condition de notre grain
    au sommet, d'abord pour savoir si il a un voisin a gauche, droite, en bas a droite ou en bas a gauche. 
    """
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
    
    """
    On verifie que nos grains au sommet n'aura pas a cause de la chute d'autres grains des voisins supplementaires.
    """
    futur_voisin_bas_droite = np.zeros(old_grille.shape, dtype=bool)
    futur_voisin_bas_droite[:-1, :-1] = masque_chute[1:, 1:]
    futur_voisin_bas_gauche = np.zeros(old_grille.shape, dtype=bool)
    futur_voisin_bas_gauche[:-1, 1:] = masque_chute[1:, :-1]
    
    """
    On combine nos masques pour avoir un ensemble de conditions sur des masques complets.
    """
    masque_droite = sommets * pas_voisin_droite * pas_voisin_bas_droite * np.logical_not(futur_voisin_bas_droite)
    masque_gauche = sommets * pas_voisin_gauche * pas_voisin_bas_gauche * np.logical_not(futur_voisin_bas_gauche)
    masque_gauche_pas_droite = masque_gauche * np.logical_not(masque_droite)
    masque_droite_pas_gauche = masque_droite * np.logical_not(masque_gauche)
    masque_droite_gauche = masque_droite * masque_gauche
    
    """
    On utilise nos masques complets a notre grille pour l'actualiser.
    """
    grille[masque_droite + masque_gauche] = 0

    rand = np.random.binomial(1, 0.5, old_grille.shape).astype(bool)
    grille[1:, 1:][(masque_droite_gauche * rand)[:-1, :-1]] = 1  # gere les chutes a droite
    grille[1:, :-1][(masque_droite_gauche * np.logical_not(rand))[:-1, 1:]] = 1  # gere les chutes a gauche

    grille[1:, 1:][masque_droite_pas_gauche[:-1, :-1]] = 1
    grille[1:, :-1][masque_gauche_pas_droite[:-1, 1:]] = 1

"Fonction qui repete en boucle l'etape de notre automate cellulaire."
def animate(i, grille, plot):
    """
    :param i: integer, nombre de fois que l'on repete l'etape
    :param grille: array, espace de notre automate cellulaire
    :param plot: plot de notre automate cellulaire
    """
    etape(grille)
    plot.set_array(grille)
    return plot,


# Main

if __name__ == "__main__":
    """
    n = 100
    grille = np.zeros((n, n))
    """
    """
    #Ecoulement
    grille[:49, 0] = 1
    for i in range(20):
        grille[50+i, i] = 2
    grille[80, 18:23] = 2
    """

    """
    #Avalanche
    pos = 50
    taille = [i for i in range(15, 0, -2)]
    grille[-2:, pos - taille[0] // 2:pos + taille[0] // 2 + 1] = 1
    for i in range(1, len(taille)):
        grille[-i*2-2:-i*2, pos-taille[i]//2:pos+taille[i]//2+1] = 1

    grille[-len(taille)*2 - 10, pos+4] = 1
    """
    """
    #Sablier
    for i in range(19,80):
        grille[i+20,i]=2
        grille[i+20,99-i]=2
    for k in range(25,70):
        for j in range(19,50):
            grille[j-19,k+2]=np.random.binomial(1, 0.5)
    grille[69:71,49:51]=0
    grille[69:71,48]=2
    grille[69:71,51]=2
    """
    
    #Galton
    grille=np.zeros((160,100))
    for i in range(25,50):
        grille[i-25,i]=2
        grille[i-25,100-i]=2
        grille[i-25,i+1:100-i]=1
    for i in range(0,20):
        grille[79:,31+2*i]=2
    for i in range(0,19):
        for j in range(0,i):
            grille[25+3*i,51-i+2*j]=2

    fig = plt.figure()
    plot = plt.imshow(grille,"afmhot")
    anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=1000, interval=40, fargs=(grille, plot), repeat=False)
    plt.show()
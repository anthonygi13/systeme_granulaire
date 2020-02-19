# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def initialisation_automate(n=100, p=0.3):
    return np.random.binomial(1, p, (n, n))


def etape(grille):

    # FIXME: une tour qui fait toute la hauteur ne s'ecroule pas

    old_grille = np.array(grille, copy=True)

    masque_chute = (old_grille[:-1, :] == 1) * (old_grille[1:, :] == 0)
    grille[:-1, :][masque_chute] = 0
    grille[1:, :][masque_chute] = 1

    sommets = old_grille.shape[0] - np.argmin(np.flip(old_grille, axis=0), axis=0)
    x = np.indices(grille.shape)[0]
    pas_voisin_droite = old_grille[:-1, 1:] == 0
    pas_voisin_gauche = old_grille[:-1, :-1] == 0
    pas_voisin_bas_droite = old_grille[1:, 1:] == 0
    pas_voisin_bas_gauche = old_grille[1:, :-1] == 0
    futur_voisin_bas_droite = np.zeros(pas_voisin_bas_droite.shape)
    futur_voisin_bas_droite[2:, :] = masque_chute[:-2, 1:]
    futur_voisin_bas_gauche = np.zeros(pas_voisin_bas_gauche.shape)
    futur_voisin_bas_gauche[2:, :] = masque_chute[:-2, :-1]

    futur_voisin_droite = np.zeros(pas_voisin_droite.shape)
    futur_voisin_droite[1:, :] = masque_chute[:-1, 1:]
    futur_voisin_gauche = np.zeros(pas_voisin_gauche.shape)
    futur_voisin_gauche[1:, :] = masque_chute[:-1, :-1]

    masque_droite = (sommets == x)[:-1, :-1] * pas_voisin_droite * pas_voisin_bas_droite * np.logical_not(futur_voisin_bas_droite) * np.logical_not(futur_voisin_droite)
    masque_gauche = (sommets == x)[:-1, 1:] * pas_voisin_gauche * pas_voisin_bas_gauche * np.logical_not(futur_voisin_bas_gauche) * np.logical_not(futur_voisin_gauche)

    masque_gauche_pas_droite = masque_gauche
    masque_gauche_pas_droite[:, :-1] *= np.logical_not(masque_droite[:, 1:])
    masque_droite_pas_gauche = masque_droite
    masque_droite_pas_gauche[:, 1:] *= np.logical_not(masque_gauche[:, :-1])
    masque_droite_gauche = masque_droite[:, 1:] * masque_gauche[:, :-1]

    grille[:-1, :-1][masque_droite] = 0
    grille[:-1, 1:][masque_gauche] = 0

    rand = np.random.binomial(1, 0.5, masque_droite_gauche.shape).astype(bool)
    grille[1:, 2:][masque_droite_gauche * rand] = 1
    grille[1:, :-2][masque_droite_gauche * np.logical_not(rand)] = 1

    grille[1:, 1:][masque_droite_pas_gauche] = 1
    grille[1:, :-1][masque_gauche_pas_droite] = 1

    print(masque_droite)
    print(masque_gauche)
    print(masque_droite_gauche)
    print(masque_gauche_pas_droite)
    print(masque_droite_pas_gauche)


def animate(i, grille, plot):
    etape(grille)
    plot.set_array(grille)
    return plot,


if __name__ == "__main__":
    #grille = initialisation_automate()
    grille = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]])
    fig = plt.figure()
    plot = plt.imshow(grille)
    anim = animation.FuncAnimation(fig, animate, frames=1, blit=False, interval=2000, fargs=(grille, plot), repeat=False)
    plt.show()

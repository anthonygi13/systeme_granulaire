#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File : automate_cellulaire.py
# Created by Anthony Giraudo and Sebastiao Clement the 03/02/2020

"""
"""

# Modules

import numpy as np
import pygame as pg
from pygame.locals import *
import time
from colour import Color
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# Functions

def draw_text(x, y, font, text):
    """
    :param x: position x
    :param y: position y
    :param font: font
    :param text: texte a afficher a l'ecran
    """
    glColor3f(1, 1, 1)  # couleur blanche
    glWindowPos2i(x, y)  # positionnement au bon endroit
    # affichage des caracteres
    for ch in text:
        glutBitmapCharacter(font, ctypes.c_int(ord(ch)))


def unitary_circle(r):
    # FIXME: bords de 0 inutiles
    d = 2*r+1
    x, y = np.indices((d, d))
    return (np.abs(np.hypot(x-(d/2), y-(d/2))-r) < 0.5).astype(int)


def entonnoir(r_base, h, angle):
    # FIXME: mieux gerer les dimensions ?
    angle = np.pi/180 * angle
    entonnoir = np.zeros((int(r_base*2+h*np.sin(angle)*2)+1, int(r_base*2+h*np.sin(angle)*2)+1, h))
    for z in range(h):
        circle = 2*unitary_circle(int(r_base + z * np.sin(angle)))
        entonnoir[entonnoir.shape[0]//2-circle.shape[0]//2:entonnoir.shape[0]//2+circle.shape[0]-circle.shape[0]//2, entonnoir.shape[1]//2-circle.shape[1]//2:entonnoir.shape[1]//2+circle.shape[1]-circle.shape[1]//2, z] = circle
    return entonnoir


class Tas:  # classe pour gerer l'evolution d'un bloc de corail
    def __init__(self, base, **kwargs):
        """
        :param base: bloc de base (numpy array 3D) remplie avec des 0 si pas de sable, 1 pour grain de sable et 2 pour
         structure solide
        :param kwargs: si une valeur est donnee a n le bloc de base est inclu dans un grand cube de cote n
        """
        base.astype(int)
        # cherche la taille n du bloc total
        if 'n' in kwargs.keys():
            n = kwargs['n']
            if not (n >= base.shape[0] and n >= base.shape[1] and n >= base.shape[2]):
                raise ValueError("n = %d trop petit par rapport a l'argument base de forme (%d, %d, %d)"%(n, base.shape[0], base.shape[1], base.shape[2]))
        else:
            n = np.amax(base.shape)

        # inclure le bloc de base dans plus grand de forme (n, n, n), ce sera la limite, la taille maximale
        self.bloc = np.zeros((n, n, n), dtype=int)
        self.bloc[(n - base.shape[0])//2: (n - base.shape[0])//2 + base.shape[0], (n - base.shape[1])//2: (n - base.shape[1])//2 + base.shape[1], :base.shape[2]] = base

    def next(self):
        """
        Joue une etape d'evolution du tas de sable
        """
        # TODO
        pass

    def draw_bloc(self):
        # construit la matrice du nombre de voisins adjacent pour optimisation
        # consiste a ne pas dessiner les grains qui sont cachees par d'autres
        A = self.bloc != 0
        voisins = np.zeros(self.bloc.shape, dtype=int)
        voisins[:-1, :, :] += A[1:, :, :]
        voisins[1:, :, :] += A[:-1, :, :]
        voisins[:, :-1, :] += A[:, 1:, :]
        voisins[:, 1:, :] += A[:, :-1, :]
        voisins[:, :, :-1] += A[:, :, 1:]
        voisins[:, :, 1:] += A[:, :, :-1]

        # couleurs des grains et des structures fixes
        colors = [Color("yellow"), Color("white")]

        centers = np.column_stack(np.where(np.logical_and(self.bloc != 0, voisins < 6)))  # coordonnees des blocs a afficher
        for center in centers:  # pour chaque bloc a afficher
            glPushMatrix()  # permet de ne deplacer qu'un objet a la fois
            glTranslatef(center[0] - self.bloc.shape[0] / 2, center[1] - self.bloc.shape[1] / 2, center[2])  # place au bon endroit le bloc
            glColor3fv(colors[self.bloc[center[0], center[1], center[2]] - 1].get_rgb())  # colorie le bloc
            glutSolidCube(1)  # dessine faces
            glColor3fv((0, 0, 0))  # couleur noire pour les arretes
            glutWireCube(1)  # dessine arretes
            glPopMatrix()  # permet de ne deplacer qu'un objet a la fois


# Main

if __name__ == "__main__":

    # parametres
    h = 20
    r = 4
    angle = 30
    pos_z = 20
    delay = 0.5  # temps en secondes entre chaque etape, a vitesse normale
    speedmin = 1/2
    speedmax = 2
    dtheta = 10  # angle de rotation en degre lors du controle de la vision avec le clavier

    # initialisation
    pg.init()
    glutInit()
    display = (1680, 1050)  # taille de la fenetre
    pg.display.set_mode(display, DOUBLEBUF | OPENGL)
    glEnable(GL_DEPTH_TEST)  # pour cubes opaques

    gluPerspective(45, (display[0] / display[1]), 0.1, 500)  # parametrage du champs de vision
    glTranslatef(0, 0, (-pos_z-h)*3)  # placement initial
    glRotatef(-50, 1, 0, 0)  # inclinaison initiale
    glRotatef(30, 0, 0, 1)  # angle de vu intial

    # config de depart
    entonnoir_struct = entonnoir(r, h, angle)
    base = np.zeros((entonnoir_struct.shape[0], entonnoir_struct.shape[1], entonnoir_struct.shape[2]+pos_z))
    base[:, :, pos_z:] = entonnoir_struct
    indices1 = np.argmax(base, axis=0)
    indices2 = base.shape[0] - 1 - np.argmax(np.flip(base, axis=0), axis=0)
    indices2[np.logical_and(indices1 == 0, indices2 == base.shape[0]-1)] = 0
    x, y, z = np.indices(base.shape)
    base[np.logical_and(np.logical_and(x > indices1, x < indices2), base == 0)] = 1

    tas = Tas(base)  # initialisation du tas de sable

    # affichage initial
    tas.draw_bloc()
    pg.display.flip()

    t0 = time.time()  # pour controler le temps entre chaque etape
    measure_t0 = False  # idem
    pause = False  # pour controler lecture/pause au clavier
    speed = 1  # rapidite
    ntour = 1  # nombre d'etapes

    while True:  # main loop
        for event in pg.event.get():
            if event.type == pg.QUIT:  # quitter si fenetre fermee
                pg.quit()
                quit()
            if event.type == pg.KEYDOWN:
                if event.key == K_SPACE:  # play/pause si espace presse
                    pause = not pause
                #if event.key == K_r:  # reset si r presse
                #    recif = Tas(np.random.binomial(1, 1 / 3, (taille_base, taille_base, 1)), lifetime, n=n)
                #    ntour = 1
                # controle de la vitesse avec p et m
                if event.key == K_p:
                    speed = min(speedmax, speed * 2)
                if event.key == K_m:
                    speed = max(speedmin, speed / 2)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # on efface tout avant de redessiner dans la fenetre

        keystate = pg.key.get_pressed()
        if keystate[K_s]:  # zoom - avec s
            glScalef(0.9, 0.9, 0.9)
        if keystate[K_z]:  # zoom + avec z
            glScalef(1.1, 1.1, 1.1)
        if keystate[K_RIGHT]:  # rotation si fleche droite
            glRotatef(dtheta, 0, 0, 1)
        if keystate[K_LEFT]:  # rotation si fleche gauche
            glRotatef(dtheta, 0, 0, -1)
        if keystate[K_UP]:  # rotation si fleche haut
            glRotatef(dtheta/2, 0, 1, 0)
        if keystate[K_DOWN]:  # rotation si fleche bas
            glRotatef(dtheta/2, 0, -1, 0)

        if not pause and delay/speed-(time.time()-t0) <= 0:  # controle le temps entre chaque etape
            ntour += 1
            tas.next()  # fait evoluer le tas de sable
            measure_t0 = True  # pour controller le temps entre chaque etape

        tas.draw_bloc()  # dessine le tas de sable

        # affichage parametres a l'ecran
        lines = []
        lines.append("Nombre de tour : {}".format(ntour))
        lines.append("Etat de la simulation : {}".format("En cours" if not pause else "En pause"))
        lines.append("Vitesse : {}".format(speed))
        linespace = 20
        x, y = 10, display[1] - 21
        for line in lines:
            draw_text(x, y, GLUT_BITMAP_9_BY_15, line)
            y -= linespace


        # affichage controles
        x, y = display[0] - 400, display[1] - 21
        lines = []
        lines.append("Play/Pause : barre d'espace")
        #lines.append("Rotation : fleches droite et gauche")
        #lines.append("Zoom : fleches haut et bas")
        lines.append("Controle de la vitesse (+/-) : touches P/M")
        #lines.append("Reinitialisation du tas de sable : R")
        for line in lines:
            draw_text(x, y, GLUT_BITMAP_9_BY_15, line)
            y -= linespace

        pg.display.flip()  # affiche tout a l'ecran

        # pour controler le temps entre chaque etape
        if measure_t0:
            t0 = time.time()
            measure_t0 = False

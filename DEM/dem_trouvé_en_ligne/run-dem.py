#!/usr/bin/python
# -*- coding: utf-8 -*-
# MIT License, Copyright (c) 2018, Damien Andre


import minidem as dem
import random


class grain:
    """grain class that represent a circular discrete element with a
 - radius :  self.rad,
 - position : self.pos
 - velocity : self.vel
 - acceleration : self.acc
 - force : self.force
 - mass : self.mass
"""

    def __init__(self, pos, radius, mass):
        self.radius = float(radius)
        self.pos = dem.vec(pos)
        self.mass = float(mass)
        self.vel = dem.vec(0., 0.)
        self.acc = dem.vec(0., 0.)
        self.force = dem.vec(0., 0.)


def contact(gr1, gr2):
    """a function that compute contact between two grains.
If the contact is detected, repulsive force are computed.
The repuslive force take into account stiffness and damping factor"""
    stiffness = 10000.
    damping = 14.
    dist = (gr2.pos - gr1.pos).get_length()
    delta = -dist + gr1.radius + gr2.radius
    normal = (gr2.pos - gr1.pos) / dist
    if (delta > 0.):
        f = normal * delta * stiffness
        gr1.force -= f
        gr2.force += f

        # manage damping factor
        diff_vel = (gr2.vel - gr1.vel) * normal
        F = damping * diff_vel * normal
        gr1.force += F
        gr2.force -= F


def time_loop():
    # get the current value of iteration
    it = dem.iteration

    # apply gravity
    for gr in dem.grain_list:
        gr.force = dem.vec(0., -9.81 * gr.mass)

    # detect contact
    for i in range(len(dem.grain_list)):
        for j in range(i + 1, len(dem.grain_list)):
            contact(dem.grain_list[i], dem.grain_list[j])

    # apply velocity verlet scheme
    dt = 0.006
    for gr in dem.grain_list:
        a = gr.force / gr.mass
        gr.vel += (gr.acc + a) * (dt / 2.)
        gr.pos += gr.vel * dt + a * (dt ** 2.) / 2.
        gr.acc = a

    # apply boundary condition
    for gr in dem.grain_list:

        if gr.pos.x - gr.radius < 0:
            gr.pos.x = gr.radius
            if gr.vel.x < 0.:
                gr.vel.x *= -.9

        elif gr.pos.x + gr.radius > 100:
            gr.pos.x = 100 - gr.radius
            if gr.vel.x > 0.:
                gr.vel.x *= -.9

        if gr.pos.y - gr.radius < 0:
            gr.pos.y = gr.radius
            if gr.vel.y < 0.:
                gr.vel.y *= -.9

        elif gr.pos.y + gr.radius > 100:
            gr.pos.y = 100 - gr.radius
            if gr.vel.y > 0.:
                gr.vel.y *= -.9


dem.init()
dem.loop_function = time_loop
dem.max_iteration = 1000

for x in range(10, 100, 10):
    for y in range(10, 100, 10):
        x_rand = x + random.random()
        y_rand = y + random.random()
        r_rand = 4 + random.random()
        dem.grain_list.append(grain((x_rand, y_rand), r_rand, 1))

dem.run()




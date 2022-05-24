#!/usr/bin/env python
# coding: utf-8


from tkinter import *

import time
from random import *
import numpy as np

WIDTH = 10
HEIGHT = 100
pix_p_m = 50  # pixels par metre
# nb_cases_w = w // pix_p_m #distance totale horizontale
# nb_cases_h = h // pix_p_m
rayon = 50
diam = 100


def proba(x):
    alpha = x % 1
    p = random()
    if p < alpha:
        return (int(x) + 1)
    else:
        return int(x)


def proba_vitesse():
    v = random()
    if v < 0.1:
        return 1
    elif v < 0.3:
        return 2
    elif v < 0.7:
        return 3
    elif v < 0.9:
        return 4
    else:
        return 5


def alea_gd(y):
    r = randint(0, 1)
    if r == 0:
        return (y + 1)
    else:
        return (y - 1)


def moyenne(t):
    n = len(t)
    m = sum(x for x in t)
    return m / n


def tri_insertion(table_humain):
    for i in range(1, len(table_humain)):
        a = table_humain[i]
        y = a.coords[1]
        j = i - 1
        while j >= 0 and table_humain[j].coords[1] > y:
            table_humain[j + 1] = table_humain[j]
            j -= 1
        table_humain[j + 1] = a
    return table_humain


def tableau():
    l = []
    for i in range(HEIGHT):
        l.append([0 for _ in range(WIDTH)])
    return l


def densite(cases):
    d = 0
    for x in range(HEIGHT):
        for y in range(WIDTH):
            if cases[x][y] == 1:
                d += 1
    return d / (WIDTH * HEIGHT)


def est_libre(x, y, cases):
    return (cases[x][y] == 0)


class Human:
    def __init__(self, x, y):
        self.coords = x, y
        self.v0 = proba_vitesse()
        self.time = 0

    def bord_gauche(self):
        x, y = self.coords
        return (y == 0)

    def bord_droit(self):
        x, y = self.coords
        return (y == (WIDTH - 1))

    def fin_marche(self):
        x, y = self.coords
        return (x == (HEIGHT - 1))

    def supprime(self, l):
        i = 0
        x, y = self.coords
        while i < len(l) and l[i].coords != (x, y):
            i += 1
        if i < len(l):
            del l[i]

    def retrouve(self, l):
        i = 0
        x, y = self.coords
        while i < len(l) and l[i].coords != (x, y):
            i += 1
        if i < len(l):
            return i
        else:
            raise Exception('camarchepas')

    def update(self):
        t = self.time
        self.time = t + 1

    def move(self, table, cases):
        i = self.retrouve(table)
        pas = self.v0
        while pas > 0:
            x, y = self.coords
            if self.fin_marche():
                cases[x][y] = 0
                self.supprime(table)
                pas = 0
            else:
                if cases[x + 1][y] == 0:
                    self.coords = x + 1, y
                    cases[x + 1][y] = 1
                    cases[x][y] = 0
                    table[i] = self
                else:
                    h2 = retrieve(x + 1, y, table)
                    if pas > h2.v0:
                        if self.bord_gauche():
                            if not (est_libre(x, y + 1, cases)):
                                pass
                            else:
                                self.coords = x, y + 1
                                cases[x][y] = 0
                                cases[x][y + 1] = 1
                                table[i] = self
                        elif self.bord_droit():
                            if not (est_libre(x, y - 1, cases)):
                                pass
                            else:
                                self.coords = x, y - 1
                                cases[x][y] = 0
                                cases[x][y - 1] = 1
                                table[i] = self
                        else:
                            if not (est_libre(x, y - 1, cases)) and not (est_libre(x, y + 1, cases)):
                                pass
                            elif not (est_libre(x, y - 1, cases)):
                                self.coords = x, y + 1
                                cases[x][y] = 0
                                cases[x][y + 1] = 1
                                table[i] = self
                            elif not (est_libre(x, y + 1, cases)):
                                self.coords = x, y - 1
                                cases[x][y] = 0
                                cases[x][y - 1] = 1
                                table[i] = self
                            else:
                                y_gd = alea_gd(y)
                                self.coords = x, y_gd
                                cases[x][y] = 0
                                cases[x][y_gd] = 1
                                table[i] = self

                pas -= 1


def retrieve(x, y, l):
    for h in l:
        if h.coords == (x, y):
            return h
    raise Exception('merde')


def flux_entrant(n, cases):
    flux = []
    while len(flux) < n:
        r = randint(0, WIDTH - 1)
        if est_libre(0, r, cases):
            h = Human(0, r)
            flux.append(h)
            cases[0][r] = 1
    return flux


def vitesse(flux):
    cases = tableau()
    table = []
    les_vitesses_moy = []
    for t in range(50):
        table.extend(flux_entrant(proba(flux), cases))
        for h in table:
            h.update()
            h.move(table,cases)
    for t in range(200):
        table.extend(flux_entrant(proba(flux), cases))
        for h in table:
            h.update()
            if h.fin_marche():
                les_vitesses_moy.append(HEIGHT / h.time)
            h.move(table,cases)
    return moyenne(les_vitesses_moy)




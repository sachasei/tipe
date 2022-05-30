#!/usr/bin/env python
# coding: utf-8


from tkinter import *
import math
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from random import *

WIDTH = 20
HEIGHT = 30
ppm = 20
N0: int = WIDTH*HEIGHT  # densité de 1personne /m^2
inf = 1e6
Cmax = 9 #nombre maximal de personnes acceptées sur 1m^2

def moyenne(t):
    n = len(t)
    m = sum(x for x in t)
    return m / n

def ecart_type(t):
    mu = moyenne(t)
    n = len(t)
    sigma = sum((x-mu)**2 for x in t) / n
    return math.sqrt(sigma)


def derivee(f):
    df_dt = []
    n = len(f)
    for t in range(n-1):
        df_dt.append(f(t+1) - f(t))
    return df_dt

def tableau(): #un environnement avec une seule sortie et sans obstacle
    l = []
    for i in range(HEIGHT-1):
        l.append([0 for _ in range(WIDTH)])
    l.append([-1 for _ in range(WIDTH)])
    l[HEIGHT-1][9] = 0
    l[HEIGHT-1][10] = 0
    l[HEIGHT-1][11] = 0

    return l

def grad(x, y, x1, y1, cases): #gradient de congestion entre la position p0 et la position souhaitable p1
    return cases[x1][y1] - cases[x][y]


def file_pref(x, y): #file de positions souhaitables a l'instant t+1, ordre décroissant
    if x == 0:
        if y == 0:
            l = [(x+1, y+1), (x+1, y), (x, y+1), (x, y)]
        elif y < 9:
            l = [(x + 1, y + 1), (x + 1, y), (x, y + 1), (x, y), (x + 1, y - 1), (x, y - 1)]
        elif y < 12:
            l = [(x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x, y), (x, y - 1), (x, y + 1)]
        elif y < WIDTH - 1:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y), (x + 1, y + 1), (x, y + 1)]
        else:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y)]
    elif x < HEIGHT - 2:
        if y == 0:
            l = [(x + 1, y+1), (x+1, y), (x, y+1), (x, y), (x-1, y+1), (x-1, y)]
        elif y < 9:
            l = [(x+1, y+1), (x+1, y), (x, y+1), (x, y), (x+1, y-1), (x-1, y+1), (x, y-1), (x-1, y), (x-1, y-1)]
        elif y < 12:
            l = [(x + 1, y), (x + 1, y+1), (x+1, y - 1), (x, y), (x, y - 1), (x, y + 1), (x-1, y), (x - 1, y-1),
                 (x - 1, y + 1)]
        elif y < WIDTH -1:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y), (x + 1, y + 1), (x - 1, y - 1), (x, y + 1), (x - 1, y),
                 (x - 1, y + 1)]
        else:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y), (x - 1, y - 1), (x - 1, y)]
    else:
        if y == 0:
            l = [(x, y+1),(x-1, y+1),  (x, y),  (x-1, y)]
        elif y < 9:
            l = [(x+1, y+1), (x, y+1), (x, y), (x-1, y+1), (x-1, y),(x, y-1), (x-1, y-1)]
        elif y < 12:
            l = [(x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x, y), (x, y - 1), (x, y + 1), (x - 1, y), (x - 1, y - 1),
                 (x - 1, y + 1)]
        elif y < WIDTH -1:
            l = [(x + 1, y - 1), (x, y - 1), (x, y), (x - 1, y - 1), (x - 1, y),(x, y+1), (x-1,y+1)]
        else:
            l = [(x, y - 1), (x, y), (x - 1, y - 1), (x - 1, y)]
    return l



def file_pref_panique(x, y): #file de positions souhaitables a l'instant t+1, ordre décroissant
    if x == 0:
        if y == 0:
            l = [(x+1, y+1), (x+1, y), (x, y+1), (x, y)]
        elif y < 9:
            l = [(x + 1, y + 1), (x + 1, y), (x, y + 1), (x, y), (x + 1, y - 1), (x, y - 1)]
        elif y < 12:
            l = [(x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x, y), (x, y - 1), (x, y + 1)]
        elif y < WIDTH - 1:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y), (x + 1, y + 1), (x, y + 1)]
        else:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y)]
    elif x < HEIGHT - 2:
        if y == 0:
            l = [(x + 1, y+1), (x+1, y), (x, y+1), (x, y)]
        elif y < 9:
            l = [(x+1, y+1), (x+1, y), (x, y+1), (x, y), (x+1, y-1), (x, y-1),]
        elif y < 12:
            l = [(x + 1, y), (x + 1, y+1), (x+1, y - 1), (x, y), (x, y - 1), (x, y + 1)]
        elif y < WIDTH -1:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y), (x + 1, y + 1), (x, y + 1)]
        else:
            l = [(x + 1, y - 1), (x + 1, y), (x, y - 1), (x, y)]
    else:
        if y == 0:
            l = [(x, y+1),  (x, y)]
        elif y < 9:
            l = [(x+1, y+1), (x, y+1), (x, y)]
        elif y < 12:
            l = [(x + 1, y), (x + 1, y + 1), (x + 1, y - 1), (x, y), (x, y - 1), (x, y + 1)]
        elif y < WIDTH -1:
            l = [(x + 1, y - 1), (x, y - 1), (x, y)]
        else:
            l = [(x, y - 1), (x, y)]
    return l


def min_grad(x,y,cases):
    file = file_pref(x,y)
    min, indice_min,  count = inf, -1, 0
    while file != [] :
        x1,y1 = file.pop(0)
        gradient = grad(x, y, x1,y1, cases)
        if gradient < min and valide(x1,y1,cases):
            min = gradient
            indice_min = count
        count += 1
    return indice_min

def choix_deplacement(x,y,cases):
    file = file_pref(x,y)
    while file != []:
        x1,y1 = file.pop(0)
        if (grad(x, y, x1, y1, cases) < 2) and valide(x1, y1, cases):
            return (x1, y1)
    file = file_pref(x,y)
    indice = min_grad(x,y,cases)
    if indice >= 0:
        x1, y1 = file[indice]
        return (x1, y1)
    else:
        return (x,y)

def choix_deplacement_panique(x,y,cases):
    file = file_pref_panique(x,y)
    while file != []:
        x1,y1 = file.pop(0)
        if valide(x1, y1, cases):
            return (x1, y1)
    return (x,y) #cas extreme: toutes les cases adjacentes sont en Cmax


def congestion(x,y,cases):
    return cases[x][y]

def est_libre(x, y, cases):
    return cases[x][y] == 0

def obstacle(x,y,cases):
    return cases[x][y] == -1

def valide(x,y,cases):
    return x >= 0 and x < HEIGHT and y >= 0 and y < WIDTH and not(obstacle(x, y, cases)) and cases[x][y] < Cmax


class Human:
    def __init__(self, x, y,etat):
        self.coords = x, y
        self.etat = etat # 0 pour calme, 1 pour panique, 2 pour retirés

    def changement_etat(self,cases,l, beta):
        if self.etat == 0:
            voisins = self.champ_de_vision(cases,l)
            while voisins != [] and self.etat == 0:
                voisin = voisins.pop()
                if voisin.etat == 1:
                    p = random()
                    if p < beta:
                        self.etat = 1



    def bord_gauche(self):
        x, y = self.coords
        return (y == 0)

    def bord_droit(self):
        x, y = self.coords
        return (y == (WIDTH - 1))

    def fin_marche(self):
        x, y = self.coords
        return x == (HEIGHT - 1) and (y == 9 or y == 10 or y == 11)

    def retrouve(self, l):
        i = 0
        while i < len(l) and l[i] != self:
            i += 1
        if i < len(l):
            return i
        else:
            raise Exception('camarchepas')

    def champ_de_vision(self, cases, l):
        x, y = self.coords
        champ = agents_occupant(x, y, cases, l)
        if valide(x+1, y+1, cases):
            champ += agents_occupant(x+1, y+1, cases, l)
        if valide(x+1, y, cases):
            champ += agents_occupant(x+1, y, cases, l)
        if valide(x+1,y-1,cases):
            champ += agents_occupant(x+1, y-1, cases, l)
        if valide(x,y + 1, cases):
            champ += agents_occupant(x, y+1, cases, l)
        if valide(x,y-1,cases):
            champ += agents_occupant(x, y-1, cases, l)
        if valide(x-1,y+1,cases):
            champ += agents_occupant(x-1, y+1,cases, l)
        if valide(x-1,y,cases):
            champ += agents_occupant(x-1, y,cases, l)
        if valide(x-1,y-1,cases):
            champ += agents_occupant(x-1, y-1,cases, l)
        return champ




    def contact_vois(self,cases):
        x,y = self.coords
        k = cases[x][y]
        if valide(x+1,y+1,cases):
            k += cases[x+1][y+1]
        if valide(x+1,y,cases):
            k +=cases[x+1][y]
        if valide(x+1,y-1,cases):
            k+=cases [x+1][y-1]
        if valide(x,y + 1, cases):
            k += cases[x][y+1]
        if valide(x,y-1,cases):
            k += cases[x][y-1]
        if valide(x-1,y+1,cases):
            k += cases[x-1][y+1]
        if valide(x-1,y,cases):
            k += cases[x-1][y]
        if valide(x-1,y-1,cases):
            k+= cases[x-1][y-1]
        # si ce sont des listes : prendre len(k)
        return (k - 1) #on ne se compte pas comme voisin

    def move_calme(self, table, cases_t, cases, retires): #faire une copie de cases dans la boucle : cases(t) -> cases(t+1)
        i = self.retrouve(table)
        x, y = self.coords
        if self.fin_marche():
            cases[x][y] -= 1
            del table[i]
            retires.append(self)

        else:
            x1, y1 = choix_deplacement(x, y, cases_t)
            if valide(x1, y1, cases):
                self.coords = x1, y1
                cases[x1][y1] = cases[x1][y1] + 1
                cases[x][y] = cases[x][y] - 1
                table[i] = self
            else:
                pass

    def move_panique(self, table, cases_t, cases, retires):
        i = self.retrouve(table)
        x, y = self.coords
        if self.fin_marche():
            cases[x][y] -= 1
            del table[i]
            retires.append(self)
        else:
            x1, y1 = choix_deplacement_panique(x, y, cases_t)
            if valide(x1, y1, cases):
                self.coords = x1, y1
                cases[x1][y1] += 1
                cases[x][y] -= 1
                table[i] = self
            else:
                pass

    def move(self,table,cases_t, cases, retires):
        if self.etat == 0:
            self.move_calme(table,cases_t, cases, retires)
        else:
            self.move_panique(table, cases_t, cases, retires)




def taux_contact(l,cases):
    n = len(l)
    p = [0 for k in range(N0)]
    for hum in l:
        k = hum.contact_vois(cases)
        p[k] += 1
    return (1/N0*N0)*sum(k*p[k] for k in range(N0))


def agents_occupant(x, y, cases, table):
    agents = []
    i = 0
    n = cases[x][y]
    while n > 0:
        if table[i].coords == (x, y):
            agents.append(table[i])
            n -= 1
        i += 1
    return agents

def transmission(l,cases,p):
    c = taux_contact(l,cases)
    return -c*math.log(1-p)



def situation(cases, proportion, N: int):
    table = []
    while len(table) < N:
        x = randint(0, HEIGHT - 2)
        y = randint(0, WIDTH - 1)
        if congestion(x, y, cases) < Cmax:
            cases[x][y] += 1
            p = random()
            if p < proportion: # proportion*N paniqués
                hum = Human(x, y,1)
                table.append(hum)
            else:
                hum = Human(x,y,0)
                table.append(hum)

    return table

def evolution(beta, proportion, N):
    cases = tableau()
    t = 0
    table = situation(cases, proportion, N)
    nb_paniques = [0]
    nb_calmes = [0]
    for h in table:
        if h.etat == 1:
            nb_paniques[0] += 1
        else:
            nb_calmes[0] += 1
    retires = []
    nb_retires = [0]
    while len(retires) < N:
        cases_t = cases.copy()
        for h in table:
            paniques, calmes = 0, 0
            h.changement_etat(cases,table,beta)
            h.move(table, cases_t, cases, retires)
            #if h.etat == 1:
             #   paniques += 1
            #else:
             #   calmes += 1
            #nb_paniques.append(paniques)
            #nb_calmes.append(calmes)
            #nb_retires.append(len(retires))

        t += 1
    return (t, nb_calmes, nb_paniques, nb_retires)


def gamma(paniques,retires):
    i = 0
    dr_dt = derivee(retires)
    while i < len(paniques) and paniques[i] > 0:
        i += 1
    gammas = []
    for j in range(i):
        gammas.append(dr_dt[j] / paniques[j])
    return moyenne(gammas)




def graphe(t,sus,inf,ret):
    times = np.linspace(0,t,1)
    plt.title('Evolution de la panique au cours du temps')
    plt.xlabel('t')
    plt.ylabel('nombre de personnes')
    plt.plot(times,sus,'green', label = 'Population Calme')
    plt.plot(times,inf,'red', label = 'Population paniquée')
    plt.plot(times,ret,'blue', label = 'Population évacuée')
    plt.ylim(0,N)



def afficher(canvas,cases): #code bon !
    for x in range (HEIGHT):
        for y in range(WIDTH):
            x_im, y_im = ppm * x, ppm * y
            if cases[x][y] == -1:
                canvas.create_rectangle(y_im,x_im,y_im +ppm,x_im +ppm, fill = 'black')
            elif cases[x][y] == 0:
                canvas.create_rectangle(y_im,x_im,y_im +ppm,x_im +ppm, fill = 'white')
            elif cases[x][y] < 4:
                canvas.create_rectangle(y_im, x_im, y_im + ppm, x_im + ppm, fill='green')
            elif cases[x][y] < 7:
                canvas.create_rectangle(y_im, x_im, y_im + ppm, x_im + ppm, fill='yellow')
            else:
                canvas.create_rectangle(y_im, x_im, y_im + ppm, x_im + ppm, fill='red')

def afficher_panique(canvas,l, cases):
    for x in range(HEIGHT):
        for y in range(WIDTH):
            x_im, y_im = ppm * x, ppm * y
            if cases[x][y] == -1:
                canvas.create_rectangle(y_im,x_im,y_im +ppm,x_im +ppm, fill = 'black')
            elif cases[x][y] == 0:
                canvas.create_rectangle(y_im,x_im,y_im +ppm,x_im +ppm, fill = 'white')
    for h in l:
        x,y = h.coords
        x_im, y_im = ppm * x, ppm * y
        if h.etat == 0:
            canvas.create_rectangle(y_im, x_im, y_im + ppm, x_im + ppm, fill='green')
        else:
            canvas.create_rectangle(y_im, x_im, y_im + ppm, x_im + ppm, fill='red')




def systeme(m,t,beta,gamma):
    s, i, r = m
    ds_dt = - beta * i * s
    di_dt = beta * i * s - gamma * i
    dr_dt = gamma * i
    return [ds_dt,di_dt,dr_dt]

def solution(beta,gamma,tf,s0,i0):
    times = np.linspace(0,tf,1e3*tf)
    sol = odeint(systeme, [s0,i0,0],times,args = (beta,gamma))

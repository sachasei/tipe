import time

import numpy as np

from simulation2panique import *

if __name__ == "__main__":
    beta = 0.9 #apres une attaque terroriste par exemple
    proportion = 0.05
    def gather_data():
        t0 = time.time()
        t_panique_de_N, t_calme_de_N = [], []
        for i in range(17):
            N = int(N0*(1 + 2.5*i/10))
            time_panique, time_calme = [], []
            for j in range(5):
                t_panique , t_calme = evolution(beta,proportion, N)[0] , evolution(0,0,N)[0]
                time_panique.append(t_panique)
                time_calme.append(t_calme)
            t_panique_de_N.append(moyenne(time_panique))
            t_calme_de_N.append(moyenne(time_calme))
        return (t_panique_de_N, t_calme_de_N)

    def plot(t_panique, t_calme):
        densités = np.linspace(1,5, len(t_panique))
        plt.plot(densités,t_panique, 'red', label="Population paniquée")
        plt.plot(densités, t_calme, 'green', label="Population calme")
        plt.xlabel('densité en personne/case')
        plt.ylabel('temps d évacuation')
        plt.legend()
        plt.show()

    t_panique, t_calme = gather_data()
    plot(t_panique, t_calme)


























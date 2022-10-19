# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:31:56 2022

@author: druth
"""

import sizedists.montecarlo as mc
import numpy as np
import matplotlib.pyplot as plt

# define the parameters used to create the child size distributions
m_picker_params = {'m_min':2,'b1':4,'b2':2.3}

# only use 1000 stochastic break-up simulations for each
n_MC = 1000

# define 4 parent bubble sizes at which to compute the child size distribution
Delta_vec = np.geomspace(0.1,100,4) 

# create the object to handle the generation of the data with which to 
dg = mc.DataGeneration(m_picker_params,n_MC=n_MC,Delta_vec=Delta_vec)

# simulate the 1000 break-ups for each Delta
dg.gen_breakups()

# use the simulated data to create the child size distrubtions
dg.gen_sizedist_data()

# plot the child size distributions for each of the four parent bubble sizes
fig,ax = plt.subplots()
for di in range(len(Delta_vec)):    
    ax.loglog(dg.dists[di].bin_centers,dg.dists[di].p,label='{:1.2f}'.format(Delta_vec[di]))    
ax.legend(title=r'$\tilde{\Delta}$')
ax.set_xlabel(r'$\tilde{\delta}$')
ax.set_ylabel(r'$\tilde{p}(\tilde{\delta};\tilde{\Delta})$')
fig.tight_layout()
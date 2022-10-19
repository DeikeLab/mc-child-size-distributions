# mc-child-size-distributions

Python code to create volumetric child size distributions following the Monte Carlo approach described in Ruth *et al.* (2022), "**Experimental observations and modelling of sub-Hinze bubble production by turbulent bubble break-up**".

A simple example of how the code can be used is given below.

```python
import sizedists.montecarlo as mc
import numpy as np
import matplotlib.pyplot as plt

# define the parameters describing the distribution of the number of bubbles formed
m_picker_params = {'m_min':2,'b1':4,'b2':2.3}

# define 4 parent bubble sizes at which to compute the child size distribution
Delta_vec = np.geomspace(0.1,100,4) 

# only use 1000 stochastic break-up simulations for each parent bubble size
n_MC = 1000

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
```

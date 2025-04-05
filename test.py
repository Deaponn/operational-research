import numpy as np

# cost function
def sphere(x):
   return sum(x**2)

def roulette_wheel_selection(p):
   c = np.cumsum(p)
   r = sum(p) * np.random.rand()
   ind = np.argwhere(r <= c)
   return ind[0][0]

# Placeholder for every individual
population = {}
# population size
npop = 20
# lower bound
varmin = -10
# upper bound
varmax = 10
# cost function
costfunc = sphere
# each inidivdual has position(chromosomes) and cost
for i in range(npop):
   population[i] = {'position': None, 'cost': None}
for i in range(npop):
   population[i]['position'] = np.random.uniform(varmin, varmax, num_var)
   population[i]['cost'] = costfunc(population[i]['position'])


#%%
import numpy.random as rd
import numpy as np
import matplotlib.pyplot as plt
a = rd.normal(2,4,30)
b = rd.normal(3,3,30)
c = rd.normal(4,4,30)
d = np.concatenate([a,np.concatenate([b,c])])
sort = sorted(d)
print(sort)
print("Variety ",np.var(sort))
print("Average ", np.mean(sort))
plt.hist(sort, bins = 10)
plt.ylabel('Amount')

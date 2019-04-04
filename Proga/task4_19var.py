#%%
import matplotlib.pyplot as plt
import numpy as np
a = 3.1
b = 20
step = 0.01
x = np.arange(a,b,step)
y = 1/( np.tan( np.sqrt( (np.log(x)-1)/(x-3) ) ) )
print("Average",np.mean(y))
plt.plot(x,y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
import math
import numpy as np
import matplotlib.pyplot as plt


def Func(x):
    return 1/(1 + np.exp(-(x-0.5)*10))

x = np.linspace(0,1,50)
plt.plot(x,Func(x))
plt.show()
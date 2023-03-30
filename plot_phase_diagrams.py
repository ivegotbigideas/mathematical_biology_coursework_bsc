from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np

# constant defaults
alpha = 1
beta = 1
gamma = 1
delta = 1

# equations
def du_dt(u,v):
    print(u)
    term_1 = alpha*u**2
    term_2 = -beta*(u*v)*(gamma + u)
    return term_1+term_2

def dv_dt(u,v):
    term_1 = v*(1-v)
    term_2 = -delta*u*v
    return term_1 + term_2

# setup plot
fig = plt.figure(figsize=(8,8))
fig.tight_layout(pad=5.0)
fig.subplots_adjust(bottom=0.3)
ax = fig.add_subplot(1,1,1)

# data preparation functions
def prepare_derivative_data(U,V):
    DU = np.zeros(U.shape)
    DV = np.zeros(U.shape)
    NI, NJ = U.shape

    for i in range(NI):
        for j in range(NJ):
            DU[i,j] = du_dt(U[i,j],V[i,j])
            DV[i,j] = dv_dt(U[i,j],V[i,j])

    clrMap = (np.hypot(DU, DV))
    clrMap[ clrMap==0 ] = 1
    DU /= clrMap
    DV /= clrMap
    return DU, DV, clrMap

# prepare data
u = np.linspace(0,10,20)
v = np.linspace(0,10,20)
U, V = np.meshgrid(u, v)
DU, DV, clrMap = prepare_derivative_data(U,V)

# plot quivers
Q = ax.quiver(U, V, DU, DV, clrMap, pivot='mid')
ax.grid()

# display
plt.show()
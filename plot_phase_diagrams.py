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
    term_1 = alpha*u**2
    term_2 = -beta*(u*v)*(gamma + u)
    return term_1+term_2

def dv_dt(u,v):
    term_1 = v*(1-v)
    term_2 = -delta*u*v
    return term_1 + term_2

# setup plot
fig = plt.figure(figsize=(5,5))
fig.tight_layout(pad=5.0)
fig.subplots_adjust(bottom=0.4)
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

def update_plot(*args):
    # update network values
    global alpha
    global beta
    global gamma
    global delta
    alpha = alpha_slider.val
    beta = beta_slider.val
    gamma = gamma_slider.val
    delta = delta_slider.val

    # update derivative data
    DU, DV, clrMap = prepare_derivative_data(U,V)
    Q.set_UVC(DU, DV)
    fig.canvas.draw()

# prepare data
u = np.linspace(0,2,20)
v = np.linspace(0,2,20)
U, V = np.meshgrid(u, v)
DU, DV, clrMap = prepare_derivative_data(U,V)

# plot quivers
Q = ax.quiver(U, V, DU, DV, pivot='mid', width=0.005, headwidth=2)
ax.grid()

# sliders
alpha_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'alpha slider', valmin=1, valmax=3, valinit=alpha, valstep=0.01)
beta_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'beta slider', valmin=1, valmax=3, valinit=beta, valstep=0.01)
gamma_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), 'gamma slider', valmin=1, valmax=3, valinit=gamma, valstep=0.01)
delta_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), 'delta slider', valmin=1, valmax=3, valinit=delta, valstep=0.01)

alpha_slider.on_changed(update_plot)
beta_slider.on_changed(update_plot)
gamma_slider.on_changed(update_plot)
delta_slider.on_changed(update_plot)


# display
plt.show()
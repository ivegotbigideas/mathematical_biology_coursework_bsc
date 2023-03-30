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
    term_1 = alpha*u**2*(1-u)
    term_2 = -beta*(u*v)/(gamma + u)
    return term_1+term_2

def dv_dt(u,v):
    term_1 = v*(1-v)
    term_2 = -delta*u*v
    return term_1 + term_2

def nullcline_1(u):
    return (1/beta)*(alpha*u*(1-u)*(gamma+u))

def nullcline_2(u):
    return 1-delta*u

# setup plot
fig = plt.figure(figsize=(5,6))
fig.subplots_adjust(bottom=0.35)
ax = fig.add_subplot(1,1,1)
ax.set_xlim(-0.05, 1.3)
ax.set_ylim(-0.05, 1.3)

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
u = np.linspace(0,1.25,30)
v = np.linspace(0,1.25,30)
U, V = np.meshgrid(u, v)
DU, DV, clrMap = prepare_derivative_data(U,V)

# plot quivers
Q = ax.quiver(U, V, DU, DV, pivot='mid', width=0.002, headwidth=3, headlength=5)

# plot nullclines
null_1, = ax.plot(u,nullcline_1(u), color="b")
null_2, = ax.plot(u,nullcline_2(u), color="orange")
null_3_vert = ax.axvline(0, color="b")
null_4_horiz = ax.axhline(0, color="orange")

# update plot function
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

    # update nullcline data
    null_1.set_data(u, nullcline_1(u))
    null_2.set_data(u, nullcline_2(u))

    fig.canvas.draw()

# sliders
alpha_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'alpha slider', valmin=0.001, valmax=3, valinit=alpha, valstep=0.01)
beta_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'beta slider', valmin=0.001, valmax=3, valinit=beta, valstep=0.01)
gamma_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), 'gamma slider', valmin=0.001, valmax=3, valinit=gamma, valstep=0.01)
delta_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), 'delta slider', valmin=0.001, valmax=3, valinit=delta, valstep=0.01)

alpha_slider.on_changed(update_plot)
beta_slider.on_changed(update_plot)
gamma_slider.on_changed(update_plot)
delta_slider.on_changed(update_plot)

# display
plt.show()
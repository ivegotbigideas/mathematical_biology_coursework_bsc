from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# constant defaults
alpha = 1
beta = 1
gamma = 1
delta = 0.75

# equations
def du_dt(u,v):
    try:
        term_1 = alpha*u**2*(1-u)
        term_2 = -beta*(u*v)/(gamma + u)
    except:
        pass
    return term_1+term_2

def dv_dt(u,v):
    term_1 = v*(1-v)
    term_2 = -delta*u*v
    return term_1 + term_2

def two_dim_system(coords):
    u = coords[0]
    v = coords[1]
    return [du_dt(u,v), dv_dt(u,v)]

def nullcline_1(u):
    return (1/beta)*(alpha*u*(1-u)*(gamma+u))

def nullcline_2(u):
    return 1-delta*u

def df_du(u,v):
    term_1 = alpha*u*(2-3*u)
    term_2 = -(beta*v*gamma)/((gamma+u)**2)
    return term_1 + term_2

def df_dv(u,v):
    return -beta*u/(gamma+u)

def dg_du(u,v):
    return -delta*v

def dg_dv(u,v):
    return 1-2*v-delta*u

def jacobian(coords):
    u = coords[0]
    v = coords[1]
    return [[df_du(u,v), df_dv(u,v)],[dg_du(u,v), dg_dv(u,v)]]

def evaluate_stability(fixed_point):
    evaluation = evaluate_fixed_point(np.array(fixed_point))

    linearisation_matrix = jacobian(fixed_point)
    
    eigenvalues,eigenvectors = np.linalg.eig(linearisation_matrix)

    stability = "unknown"
    if any(np.real(eigenvalue) == 0 for eigenvalue in eigenvalues):
        stability = "unknown"
    elif all(np.real(eigenvalue) < 0 for eigenvalue in eigenvalues):
        stability = "stable"
    elif any(np.real(eigenvalue) > 0 for eigenvalue in eigenvalues):
        stability = "unstable"
    
    print(fixed_point)
    print(eigenvalues)
    print("\n")
    return stability

def evaluate_fixed_point(fp):
    evaluation = two_dim_system(fp)
    return evaluation

def norm_of_evaluated_point(point):
    return np.linalg.norm(evaluate_fixed_point(point))

def find_fixed_points():
    starting_guesses = np.random.uniform(low=0,high=5, size=(49,2))
    starting_guesses = starting_guesses.tolist()
    starting_guesses.append([0]*2)
    starting_guesses.append([1,0])
    starting_guesses.append([0,1])
    fixed_points = []
    for guess in starting_guesses:
        fixed_point = sp.optimize.root(two_dim_system, guess, tol=1e-12, method='hybr', jac=jacobian)

        add_new_fixed_point = True
        for existing_fp in fixed_points:
            if (norm_of_evaluated_point(existing_fp - fixed_point.x) < 0.005) and (list(fixed_point.x) != [0]*2) and (list(fixed_point.x) != [0,1]) and (list(fixed_point.x) != [1,0]):
                add_new_fixed_point = False

        if norm_of_evaluated_point(fixed_point.x) > 1e-12:
            add_new_fixed_point = False

        if add_new_fixed_point:
            fixed_points.append(fixed_point.x)

    fixed_points = list(set(tuple(row) for row in fixed_points))
    return fixed_points

# setup plot
fig = plt.figure(figsize=(6,6))
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

# plot fixed points function
def plot_fixed_points(*args):
    fixed_points = find_fixed_points()
    C = []
    for point in fixed_points:
        stability = evaluate_stability(point)
        if stability == "stable":
            C.append(ax.plot(point[0], point[1],"red", marker = "x", markersize = 7.0))
        elif stability == "unstable":
            C.append(ax.plot(point[0], point[1],"red", marker = "o", markersize = 7.0))
        else:
            C.append(ax.plot(point[0], point[1],"red", marker = "^", markersize = 7.0))

# prepare data
u = np.linspace(0,1.25,30)
v = np.linspace(0,1.25,30)
U, V = np.meshgrid(u, v)
DU, DV, clrMap = prepare_derivative_data(U,V)

# plot quivers
Q = ax.quiver(U, V, DU, DV, pivot='mid', width=0.002, headwidth=3, headlength=5, zorder=5)

# plot nullclines
null_1, = ax.plot(u,nullcline_1(u), color="cyan", zorder=-5)
null_2, = ax.plot(u,nullcline_2(u), color="orange", zorder=-5)
null_3_vert = ax.axvline(0, color="cyan", zorder=-5)
null_4_horiz = ax.axhline(0, color="orange", zorder=-5)

# plot fixed points
plot_fixed_points()

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
alpha_slider = Slider(plt.axes([0.25, 0.1, 0.65, 0.03]), 'alpha slider', valmin=0, valmax=3, valinit=alpha, valstep=0.01)
beta_slider = Slider(plt.axes([0.25, 0.15, 0.65, 0.03]), 'beta slider', valmin=0, valmax=3, valinit=beta, valstep=0.01)
gamma_slider = Slider(plt.axes([0.25, 0.2, 0.65, 0.03]), 'gamma slider', valmin=0, valmax=3, valinit=gamma, valstep=0.01)
delta_slider = Slider(plt.axes([0.25, 0.25, 0.65, 0.03]), 'delta slider', valmin=0, valmax=3, valinit=delta, valstep=0.01)

alpha_slider.on_changed(update_plot)
beta_slider.on_changed(update_plot)
gamma_slider.on_changed(update_plot)
delta_slider.on_changed(update_plot)

# button
ax_fp = fig.add_axes([0.81, 0.01, 0.1, 0.075])
fp_btn = Button(ax_fp, 'FPs')
fp_btn.on_clicked(plot_fixed_points)

# display
plt.show()

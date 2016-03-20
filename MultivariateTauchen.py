
# coding: utf-8

# In[1]:

import numpy as np
import operator
import functools
import scipy 
from scipy.stats import norm
import itertools


def multidimensional_tauchen(A, Sigma, m = 3, n = 7):
    """
    [y, P] = multidimensional_tauchen(A, Sigma, m = 3, n = 7)
    
    *******************************************************************
    *************************** BACKGROUND ****************************
    *******************************************************************
    Using the process described in Tauchen (86), we want to approximate
    the continuous system,
    
                          Y_t = A Y_{t-1} + e_t       (*)
                          
    with a discretized version. Here Y_t, e_t are a Mx1 vectors.
    
    The random vector, e_t ~ N(0,Sigma), is assumed to be IID in the time 
    domain and jointly normal across its individual components e_{i,t}. This
    assumption implies that the transformed vector,
    
                            ebar_t = Q'e_t,  
    
    where Q is an orthogonal matrix of eigenvectors for Sigma, is 
    IID Normal and mutually independent across its individual components, 
    ebar_{i,t}. i.e.) ebar_t ~ N(0,Gamma), where Gamma is the 
    diagonal matrix of eigenvalues, Gamma_i, correpsonding to Q, and 
    ebar_{i,t} ~ N(0,Gamma_i).
    
    The rationale behind this is that the transformed system,
    
                     Q'Y_t = Q'AQ(Q'Y_{t-1}) + Q'e_t, 
                                or
                    Ybar_t = Abar Ybar_{t-1} + ebar_t,      (**)
                            
    satisfies the assumptions made in Tauchen (1986) that the components of 
    ebar_t be mutually independent and IID over time. We then apply 
    Tauchen's proposed method to the linearly transformed system in (**), 
    noting that the two systems are related by the orthogonal matrix, Q,(i.e.
    QQ' = I, which implies that QYbar_t = QQ'Y_t = Y_t).
    
    This allows for a lot of generality in the original system. The components
    of Y_t can be related through non-zero offdiagonal terms in Sigma (corr.
    shocks), in A (dependence on lagged variables elsewhere in the system), or
    in both.
    
    Roadmap:
    (a) Follow Tauchen's method to approximate the continuous-valued system 
        in (**) with one that is discrete-valued.
        --> Outcome: A N*-state system taking values in the grid 
                     Ybar_1 < ... < Ybar_N*
    (b) Form the transition matrix, P, for the discrete-system developed in (1)
    (c) Vectorizing the grid from (a), Ybar = [Ybar_1, ...,  Ybar_N*], we can
        then form the state space for the original system (Y) by premultipying 
        Ybar by Q (since Ybar_j approximates Ybar_t, Ybar_t = Q'Y_t, QQ' = I).
        The transition matrix from (b) remains unchanged.

    
    *******************************************************************
    *************************** PARAMETERS ****************************
    *******************************************************************

    A     : MxM matrix(float)
          Matrix of autoregressive parameters
    Sigma : MxM symmetric matrix(float)
          Covariance matrix for e_t
    m     : scalar(int)
          The number of standard deviations to approximate out to
    n     : scalar(int)
          The number of states per ind. component of Y_t to use in 
          the approximation. There will be n^M total states, with M
          denoting the # of components of Y_t. It is assumed that
          each component takes on the same # of states.
          
        
    *******************************************************************
    ***************************** Returns *****************************
    *******************************************************************
    
    y : matrix(float, dim= (n^M)xM )  
        The state space of the multidimensional discretized process. The
        'M' columns of y correspond to the state spaces of the individual
        components of Y_t.
    
    P : array_like(float, ndim= (n^M)x(n^M))
        The multidimensional Markov transition matrix where P[i, j] 
        is the probability of the system transitioning from y[i] to y[j].
         
         
    *******************************************************************
    ************************** PRELIMINARIES **************************
    *******************************************************************
    """
    # Verify that A and Sigma are both numpy matrices. 
    A, Sigma = np.matrix(A), np.matrix(Sigma)
    
    # Verify that Sigma is symmetric.
    if (Sigma.T == Sigma).all() == False:
        print("Sigma must be a symmetric matrix or array")
    
    # Denote the number of components of Y_t by M
    M = len(Sigma)
    
    # Total number of states in the system = n^M
    Nstar = n**M 
    
    
    """
    *******************************************************************
    ************** STEP 1: FORM THE TRANSFORMED SYSTEM: ***************
    **************  Ybar_t = Abar Ybar_{t-1} + ebar_t   ***************
    *******************************************************************
    """
    # Eigenvalue decomposition of Sigma
    eigvals, Q = np.linalg.eigh(Sigma)
    Gamma = np.diag(eigvals)
 
    # Transformed matrix of autoregressive parameters
    A_bar = Q.T*A*Q
    
    
    """
    *******************************************************************
    ******* STEP 2: Create the multidimensional grid for Ybar_t *******
    *******************************************************************
    """    
    # NOTE: We denote the grid for Ybar_t with x's and Y_t with y's.

    # Unconditional Covariance Matrix of Ybar
    uncon_covar_bar = scipy.linalg.solve_discrete_lyapunov(A_bar,Gamma)

    # Create upper and lower grid bounds for each component of Ybar.
    x_max = [m*np.sqrt(uncon_covar_bar[i,i]) for i in range(M)]
    x_min = [-1*x_max[i] for i in range(M)]

    # Create the discrete grid for each ind. component of Ybar
    x = []
    for i in range(M):
        x.append(np.linspace(x_min[i],x_max[i],n).tolist())

    # Create the index for the multidimensional grid
    MD_grid_index = list(itertools.product(range(n), repeat=M))
    
    # NOTE: We denote the grid for Ybar_t with x's and Y_t with y's.
    
    # Next we create the multidimensional grid.

    MD_grid = []
    for i in range(Nstar):
        MD_grid.append([x[j][MD_grid_index[i][j]] for j in range(M)])
    
    
    """
    *******************************************************************
    ***** STEP 3: Create the multidimensional transition matrix, P ****
    *******************************************************************
    Note: Per the Tauchen paper, we first create kernels h_i(j,l) for 
    each ind. component Ybar_i of the Mx1 vector Ybar. Each h_i has 
    dimension = (Nstar,n) with 
    h_i(j,l)=P(Ybar_{i,t} in l | Ybar_{t-1} in j)
    
    The code to create each h_i is almost identical to the Quant-Econ 
    univariate Tauchen code with mu[j][a] here replacing rho*x[j]. Note 
    that mu_{j} = [mu_{1,j}, ..., mu_{M,j}]'=Abar*Ybar_t when Ybar_t is 
    in state j.
    
    The process for creating transition matrix, P, is as follows: 
    
    Let L_i(k) be the state of the ith component of Ybar_t when Ybar_t 
    is in state k. 
    
    e.g.) if Ybar_t = [Ybar_{1,t},Ybar_{2,t}]' is in state k = [a,b], 
    then L_1(k) = a  and  L_2(k) = b. 
    
    By mutual independence of the Ybar_{i,t}'s we then have,
    
      P(j,k) = h_1[j,L_1(k)] * ... * h_M[j,L_M(k)]  j,k = 1, ... Nstar
    *******************************************************************
    """
    # mu =  Abar*Ybar_t
    mu= [np.dot(np.array(A_bar),np.array(MD_grid[i]).T) for i in range(Nstar)]

    # Form CDFs for each of the mutually independent components of ebar_t
    F = [norm(loc=0, scale = np.sqrt(eigen)).cdf for eigen in eigvals]

    # Create the kernels, h_i, for i = 1, ..., M
    h = []
    step = [(x_max[i] - x_min[i]) / (n - 1) for i in range(M)]
    half_step = [0.5 * step[i] for i in range(M)]

    for a in range(M):
        h.append(np.empty((Nstar, n)))
        for j in range(Nstar):
            h[a][j,0] = F[a](x[a][0] - mu[j][a] + half_step[a])
            h[a][j,n-1] = 1- F[a](x[a][n-1] - mu[j][a] - half_step[a])
            for i in range(1, n-1):
                z = x[a][i] - mu[j][a]
                h[a][j, i] = F[a](z + half_step[a]) - F[a](z - half_step[a])

    # Create the transition matrix, P.
    P = np.empty((Nstar, Nstar))
    for j in range(Nstar):
        for k in range(Nstar):
            L = [MD_grid_index[k][l] for l in range(M)]
            h_j = [h[m][j,L[m]] for m in range(M)]
            P[j,k] = functools.reduce(operator.mul, h_j, 1)

    """
    *******************************************************************
    **** STEP 5: Create the multidim. grid for the original system ****
    *******************************************************************
    """
    # y = QYbar, where Ybar is the discrete grid for the system in (**) 

    y = Q*np.matrix(MD_grid).T
    y = y.T   # To maintain shape consistency with P.

    return y, P


# In[2]:

"""
*******************************************************************
*************************** Simulation ****************************
*******************************************************************
"""
import numpy as np
import matplotlib.pyplot as plt
import quantecon as qe
import pandas as pd
import statsmodels.formula.api as smf

# Parameters for the simulation
A = np.matrix([[0.945,0],[0,0.945]])
stdev = 0.025 # Square root of variance of innovation terms
rho = 0.015   # Square root of covar. of innovation terms
Sigma = np.matrix([[stdev**2,rho**2],[rho**2,stdev**2]])
m = 3
n = 21

# Generate the discretized grid/transition kernel
logy, pv = multidimensional_tauchen(A, Sigma, m ,n)
y = np.exp(logy)

# Initialize the Markov Chain somwhere around the mean
# of the discretized grid (in this case the ind. grids 
# are symmetric, so the mean of one will be the mean of the
# other.)
b = np.array(y.mean(axis=1)).T
c = y.mean(axis = 0)
y_init = np.searchsorted(b[0], c[0,0])

# Simulate 250 observations
mc = qe.markov.MarkovChain(pv) 
y_sim_indices = mc.simulate(250, init=y_init)
y_vec = y[y_sim_indices]


# In[3]:

# Plot the two similated processes 
fig, ax = plt.subplots(figsize=(10, 4))
p_args = {'lw': 2, 'alpha': 0.7}
fig.subplots_adjust(hspace=0.3)
ax.plot(y_vec[:,0], **p_args, label = 'Economy 1')
ax.plot(y_vec[:,1], **p_args, label = 'Economy 2')
ax.set_title("Simulated Processes")
ax.legend(loc =1)


# In[4]:

# Regression results for simulated process 1 (2 is similar)
# Want: Y_i,t = 0.945*Y_i,t-1
yt = y_vec[:,0].T.tolist()
output = pd.DataFrame({'y':yt[0]})
output = pd.DataFrame({'y':yt[0],'yl':output.shift()['y']})
model = smf.ols(formula='y ~ yl', data=output).fit()
print(model.summary())


# In[5]:

# Histogram of estimates for lambda over 300 iterations of the regression above
ybeta = []
for i in range(300):
    i
    mc = qe.markov.MarkovChain(pv) 
    y_sim_indices = mc.simulate(250, init=y_init)
    y_vec = logy[y_sim_indices]
    yt = y_vec[:,1].T.tolist()
    output = pd.DataFrame({'y':yt[0]})
    output = pd.DataFrame({'y':yt[0],'yl':output.shift()['y']})
    model = smf.ols(formula='y ~ yl', data=output).fit()
    ybeta.append(model.params['yl'])

plt.hist(ybeta)
print('The average estimate of lambda over 300 iterations is',np.array(ybeta).mean(),
      '\n with a standard deviation of', np.array(ybeta).std())



"""
Description:
Prob 1

Date:
2.19.2023

Author: Hunter Akins

Institution: Scripps Institution of Oceanography, UC San Diego
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import cvxpy as cp
X = cp.Variable((4,4))
B = np.ones((4,4))
B[0,:] = 0
F = 5*np.diag(np.ones(4))
print(F)

prob = cp.Problem(cp.Minimize(cp.trace(B@X)), [X >= 0])

prob.solve()
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(X.value)


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

A = np.array([[2,0,1,1],[0,0,0,1],[0,2,1,0],[2,3,-1,2],[0,2,2,0]])
c = np.array([[-1],[-2],[-3],[-1]])
b = np.array([[2],[1],[3],[5],[5]])
Apos = np.array([[2,0,1,1],[0,0,0,1],[0,2,1,0],[0,2,2,0]])
bpos = np.array([[2],[1],[3],[5]])
x = np.array([[-1/2], [-1], [1],[ 0]])


print(A@x, b)
print(A)
print(A@c)

U, s, VH = np.linalg.svd(A)
rangeA = U[:,:4]
print('U', U)
print('s', s)
print('VH', VH)
print('b', b)

print('b proj onto range A', rangeA@rangeA.T@ b)
ret = np.linalg.lstsq(A, b)
xsol = ret[0]
print('b , A@xsol', b, A@xsol)
for alpha in range(5):
    u_last = U[:,-1] # last left singular vector
    tmp1 = VH@c
    c_pseudo_inverse = -U[:,:4]@np.diag(1/s) @ tmp1
    lam = c_pseudo_inverse[:,0]+ alpha * U[:,-1]
    print('constraint', (A.T @ lam)[:,np.newaxis] + c)

print('c_pi', c_pseudo_inverse)
print('A at c_pi - c', A.T@c_pseudo_inverse + c)
print('u5', U[:,-1])

print('-b^T c_pi', b.shape, c_pseudo_inverse.shape, -b.T @ c_pseudo_inverse, 'b^T u5', -b.T@u_last)

import cvxpy as cp
x = cp.Variable(4)

prob = cp.Problem(cp.Minimize(c.T@x), [A@x <= b[:,0], x >= 0])
prob.solve()
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)
print('Dual optimal', -b.T @ prob.constraints[0].dual_value)

probpos = cp.Problem(cp.Minimize(c.T@x), [Apos@x <= bpos[:,0]])
probpos.solve()
print("A pos solution x is")
print(x.value)
print('A pos value is ', probpos.value)
print('A pos c', Apos.T@c)


lam = cp.Variable(5)
prob = cp.Problem(cp.Maximize(-b.T@lam), [A.T@lam ==-c[:,0], lam >= 0])
prob.solve()
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)


# Print result.

xstart = np.array([0, .5, 2, 0])
print('Axopt', A@xstart, 'b', b, 'diff', A@xstart - b[:,0])


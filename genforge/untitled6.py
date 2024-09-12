# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 13:11:23 2024

@author: Maisam
"""
import numpy as np

xx = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[17,18,19],[20,21,22]])
w = np.array([[ 0.7,  0.2, -0.3],
       [ 0.1,  0.02, -0.8],
       [ 0.6 ,  0.08, -0.4],
       [ 0.6,  0.5, -0.3],
       [ 0.2, -0.9, -0.6]])
b = np.array([[-10 ],
       [-0.4],
       [ 0.2],
       [ 0.8 ],
       [-0.5 ]])
zt = np.zeros((xx.shape[0], w.shape[0]))
for i in range(xx.shape[0]):
    for c in range(w.shape[0]):
        for j in range(xx.shape[1]):
            zt[i,c] = zt[i,c] + w[c,j] * xx[i,j]
        zt[i,c] = zt[i,c] + b[c]
        
expzt = np.zeros_like(zt)
for i in range(expzt.shape[0]):
    for j in range(expzt.shape[1]):
        expzt[i,j] = np.exp(zt[i,j] - np.max(zt[:,j]))

probt = np.zeros_like(expzt)
for i in range(expzt.shape[0]):
    for j in range(expzt.shape[1]):
        probt[i,j] = expzt[i,j]/np.sum(expzt[i,:])
        
z1 = np.dot(w, xx.T) + b
z = z1.T
print('')
print('z - zt = ', np.sum(np.sum(z - zt)))

exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Stability improvement
print('exp_z - exp_zt = ', np.sum(np.sum(exp_z - expzt)))

prob = exp_z / np.sum(exp_z, axis=1, keepdims=True)
print('prob - probt = ', np.sum(np.sum(prob - probt)))




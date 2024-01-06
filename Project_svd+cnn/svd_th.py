import numpy as np


A=[[1,2,3],[4,5,6],[7,8,9]]



n_components = 1

# Perform Full SVD
U, S, VT = np.linalg.svd(A)
# U - матриця лівих сингулярних векторів
# S - масив сингулярних значень
# VT - транспонування матриці правих сингулярних векторів (V)
print(U,S,VT)



# Select the top n_components singular values and vectors
U_reduced = U[:, :n_components]
S_reduced = np.diag(S[:n_components])
VT_reduced = VT[:n_components, :]

# Reconstruct the approximated matrix
A_approximated = np.dot(U_reduced, np.dot(S_reduced, VT_reduced))

print(A_approximated)

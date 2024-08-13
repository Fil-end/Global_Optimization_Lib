import numpy as np

data = np.load('asop.npz', allow_pickle = True)
miu = data['miu']
A_matrix = data['A_matrix']
n_Ag = data['Ag']
n_O = data['O']

print(n_Ag)

n_idx = 0
matrix_list = []
for idx in range(len(n_Ag)):
    Ag = n_Ag[idx]
    O = n_O[idx]
    matrix = A_matrix[n_idx]
    matrix_list.append(matrix)
    if Ag == O and Ag == np.cross(matrix[0], matrix[1]):
        n_idx += 1
# print(matrix_list, miu)
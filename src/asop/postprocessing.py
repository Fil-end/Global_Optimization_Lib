import numpy as np
import os
from lasp import LASP

from ase.build import fcc110, bulk
from ase.io import read
from asop import ASOP

K = 8.617 * 1E-5

initial_slab = fcc110('Ag', size=(20, 20, 6), vacuum=10.0)
mock_slab = fcc110('Ag', size = (1,1,6), vacuum = 10.0)
Ag_bulk = read('Ag.poscar')
Ag2O = read('Ag2O.poscar')

def get_surface_energy(TM_2D):
    primitive_cell = mock_slab.cell
    atoms = initial_slab.copy()
    # move the atoms to the center of the slab
    asop.rectify_position(atoms)

    TM_3D = np.array([[TM_2D[0][0], TM_2D[0][1], 0],
                            [TM_2D[1][0], TM_2D[1][1], 0],
                            [0                , 0                , 1]])
    atoms.set_cell(np.dot(TM_3D, primitive_cell))
    asop.in_cell(atoms)

    atoms,initial_energy =asop.initial_energy(atoms)
    return initial_energy

def list_dir(file_path):
    list = []
    dir_list = os.listdir(file_path)
    for cur_file in dir_list:
        path = os.path.join(file_path, cur_file)
        if os.path.isdir(path):
            list.append(path)
    return list

def lasp_single_calc(atoms):
    atoms.calc = LASP(task='single-energy', pot='AgO', potential='NN D3', command = 'mpirun lasp')
    energy = atoms.get_potential_energy()
    return energy 

# self.E_O = ((-0.314 - -0.323) * 2 + -1.02 + K * self.T * K * self.T * np.log(self.P / 101325)) / 2
# self.E_O = ((-20.0995 / 2 - (-26.8121) / 2) * 2 + -1.02 + K * self.T * K * self.T * np.log(self.P / 101325)) / 2
H_Ag2O = lasp_single_calc(Ag2O) / int(len(Ag2O) / 3)
H_Ag = lasp_single_calc(Ag_bulk) / len(Ag_bulk)
print(f"H_Ag2O = {H_Ag2O}, H_Ag = {H_Ag}")
Hfm_Ag2O = -0.323   # eV
E_O = ((H_Ag2O - 2 * H_Ag - Hfm_Ag2O) * 2 + -1.02 + K * 473.15 * np.log(0.25 / 101325))/ 2
E_Ag = lasp_single_calc(mock_slab) / len(mock_slab)
print(f"E_O is {E_O}, E_Ag is {E_Ag}")

if __name__ == '__main__':
    asop = ASOP(initial_slab = initial_slab,
                mock_slab=mock_slab,
                calculator_method = 'LASP',
                model_path = 'AgO')
    
    # Generate data
    dir_list = list_dir('./')

    Ag_list = []
    O_list = []
    A_matrix_list = []
    A_list = []
    grids_list = []
    energy_list = []
    initial_energy_list = []

    current_TM = None
    current_TM_index = -1
    current_initial_energy = 0
    
    for dir in dir_list:
        if os.path.exists(f'{dir}/asop.npz',):
            data = np.load(f'{dir}/asop.npz', allow_pickle = True)

            i_e_list = []

            miu = data['miu']
            Ag = data['Ag']
            O = data['O']
            A_matrix = data['A_matrix']
            A = data['A']
            grids = data['grids']
            energy = data['energy']

            for matrix in A_matrix:
                matrix = matrix.tolist()
                if matrix != current_TM:
                    current_TM = matrix
                    current_initial_energy = get_surface_energy(current_TM)
                    i_e_list.append(current_initial_energy)
                else:
                    i_e_list.append(current_initial_energy)

            Ag_list.extend(Ag)
            O_list.extend(O)
            A_matrix_list.extend(A_matrix)
            A_list.extend(A)
            grids_list.extend(grids)
            energy_list.extend(energy - np.array(i_e_list))

    miu_list = (np.array(energy_list) - E_O * np.array(O_list) - E_Ag * np.array(Ag_list)) / np.array(A_list)
    miu_list = miu_list.tolist()

    save_path = os.path.join('./', f"{'asop'}.npz")
    np.savez_compressed(
        save_path,
        miu = miu_list,
        energy = energy_list,
        Ag = Ag_list,
        O = O_list,
        A_matrix = A_matrix_list,
        A = A_list,
        grids = grids_list,
    )

    for min_idx in range(len(miu_list)):
        if miu_list[min_idx] < -0.4:
            # min_idx = miu_list.index(min(miu_list))
            n_Ag = Ag_list[min_idx]
            n_O = O_list[min_idx]
            TM = A_matrix_list[min_idx]
            A = A_list[min_idx]

            # print(miu_list)
            print(f"Logging Info: TM is {TM}, Ag is {n_Ag}, O is {n_O}, miu is {miu_list[min_idx]}")
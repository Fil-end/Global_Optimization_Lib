import numpy as np
import matplotlib.pyplot as plt
import os
from lasp import LASP

from ase import Atoms
from ase.build import fcc100
from ase.geometry.analysis import Analysis
from ase.io import read, write
from ase.io.dmol import read_dmol_arc, write_dmol_arc
from asop import ASOP

K = 8.617 * 1E-5

# initial_slab = fcc110('Ag', size=(20, 20, 6), vacuum=10.0)
mock_slab = fcc100('Ag', size = (1,1,4), vacuum = 10.0)

EV_2_J = mock_slab.cell[0][0] * mock_slab.cell[1][1] / 16.02176634

Ag_bulk = read('Ag.poscar')
Ag2O = read('Ag2O.poscar')

def env_O2(slab:Atoms) -> Atoms:
    ana = Analysis(slab)
    OOBonds = ana.get_bonds('O', 'O', unique = True)
    PdOBonds = ana.get_bonds('Ag', 'O', unique=True)

    Pd_O_list = []
    del_list = []

    if PdOBonds[0]:
        for PdO_bond in PdOBonds[0]:
            Pd_O_list.extend([PdO_bond[0], PdO_bond[1]])
    
    if OOBonds[0]:
        for OO in OOBonds[0]:
            if OO[0] not in Pd_O_list and OO[1] not in Pd_O_list:
                del_list.extend([OO[0], OO[1]])

    return bool(del_list)

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
    return atoms, initial_energy

def list_dir(file_path):
    list = []
    dir_list = os.listdir(file_path)
    for cur_file in dir_list:
        path = os.path.join(file_path, cur_file)
        if os.path.isdir(path):
            list.append(path)
    return list

def lasp_single_calc(atoms):
    atoms.calc = LASP(task='single-energy', pot='AgCHO_pf.pot', potential='NN D3', command = 'mpirun lasp')
    energy = atoms.get_potential_energy()
    return energy 

# H_Ag2O = lasp_single_calc(Ag2O) / int(len(Ag2O) / 3)
# H_Ag = lasp_single_calc(Ag_bulk) / len(Ag_bulk)
# print(f"H_Ag2O = {H_Ag2O}, H_Ag = {H_Ag}")
# Hfm_Ag2O = -0.323   # eV
# E_O = ((-11.63106265 - 2 * H_Ag - Hfm_Ag2O) * 2 + -1.08 + K * 500 * np.log(0.60 / 101325))/ 2
# E_Ag = lasp_single_calc(mock_slab) / len(mock_slab)
# print(f"E_O is {E_O}, E_Ag is {E_Ag}")

if __name__ == '__main__':
    # asop = ASOP(initial_slab = initial_slab,
    #             mock_slab=mock_slab,
    #             calculator_method = 'LASP',
    #             model_path = 'AgO')
    
    # Generate data
    dir_list = list_dir('./')

    Ag_list = []
    O_list = []
    A_matrix_list = []
    A_list = []
    grids_list = []
    energy_list = []
    miu_list = []
    arc_path_list = []
    initial_energy_list = []

    current_TM = None
    current_TM_index = -1
    current_initial_energy = 0
    
    for dir in dir_list:
        if os.path.exists(f'{dir}/asop.npz',):
            data = np.load(f'{dir}/asop.npz', allow_pickle = True)

            # i_e_list = []

            miu = data['miu'] * EV_2_J
            Ag = data['Ag']
            O = data['O']
            A_matrix = data['A_matrix']
            A = data['A']
            grids = data['grids']
            energy = data['energy']

            for idx in range(len(A_matrix)):
                n_Ag = Ag[idx]
                n_O = O[idx]
                TM = A_matrix[idx]

                u_p,v_p = TM
                norm_u = np.round(np.linalg.norm(u_p), 3)
                norm_v = np.round(np.linalg.norm(v_p), 3)
                
                arc_path = f"./{dir}/save_dir/{norm_u}x{norm_v}/Ag{n_Ag}O{n_O}/outstr.arc"
                arc_path_list.append(arc_path)

            Ag_list.extend(Ag)
            O_list.extend(O)
            A_matrix_list.extend(A_matrix)
            A_list.extend(A)
            grids_list.extend(grids)
            energy_list.extend(energy)
            miu_list.extend(miu)

    # miu_list = (np.array(energy_list) - E_O * np.array(O_list) - E_Ag * np.array(Ag_list)) / np.array(A_list)
    # miu_list = miu_list.tolist()

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
    
    # data = np.load(f'asop_1.npz', allow_pickle = True)
    target_atoms = []
    target_miu = []
    target_TM = []
    target_A = []
    target_Ag = []
    target_O = []
    # miu_list = data['miu'].tolist() * EV_2_J
    # A_matrix_list = data['A_matrix'].tolist()
    # A_list = data['A']
    # Ag_list = data['Ag']
    # O_list = data['O']
    
    for miu_idx in range(len(miu_list)):
    # miu_idx = miu_list.index(min(miu_list))
        n_Ag = Ag_list[miu_idx]
        n_O = O_list[miu_idx]
        TM = A_matrix_list[miu_idx]
        A = A_list[miu_idx]
        A_matrix = A_matrix_list[miu_idx]
        arc_path = arc_path_list[miu_idx]
        
        if miu_list[miu_idx] > -1.0:
            atoms = read_dmol_arc(arc_path)
            target_atoms.append(atoms)
            target_miu.append(miu_list[miu_idx])
            target_A.append(A)
            target_TM.append(TM)
            target_O.append(n_O)
            target_Ag.append(n_Ag)
            print(f"Logging Info: TM is {TM}, Ag is {n_Ag}, O is {n_O}, miu is {miu_list[miu_idx]}", flush=True)
        
    # np.savez_compressed(
    #     os.path.join('./', f"{'asop_1'}.npz"),
    #     miu = miu_list,
    #     Ag = Ag_list,
    #     O = O_list,
    #     A_matrix = A_matrix_list,
    #     A = A_list,
    # )
    # sort target atoms
    atoms_order = np.argsort(np.array(target_miu))
    A_matrix_list = np.array(target_TM)[atoms_order].tolist()
    A_list = np.array(target_A)[atoms_order].tolist()
    Ag_list = np.array(target_Ag)[atoms_order].tolist()
    O_list = np.array(target_O)[atoms_order].tolist()
    target_miu.sort()

    ordered_atoms = []
    for order in atoms_order:
        ordered_atoms.append(target_atoms[order])

    write('best.xyz', ordered_atoms)
    write_dmol_arc('best_traj.arc', ordered_atoms)
    
    # 创建示例数据
    A = A_list[0]
    x = np.linspace(1, A, A) / A
    y = np.linspace(1, A, A) / A
    best_data = np.random.rand(A, A)  # 生成一个10x10的随机数据矩阵
    for idx in range(len(target_miu)):
        if A_matrix_list[idx] == A_matrix_list[0]:
            n_Ag = Ag_list[idx]
            n_O = O_list[idx]
            best_data[n_Ag - 1][n_O - 1] = target_miu[idx]
    
    # 创建值热力图
    import matplotlib as mpl
    # norm1 = mpl.colors.Normalize(vmin=-0.4, vmax=0.3)
    # im1 = mpl.cm.ScalarMappable(norm=norm1, cmap='jet')
    plt.contourf(x, y, best_data, cmap='jet', levels=200, vmin = -0.2, vmax = 0.2)
    # cbar = plt.colorbar(im1,label='γ')  # 添加颜色条
    # cbar.set_ticklabels(['-0.40','-0.30','-0.20','-0.10','-0.00','0.10','0.20','≥0.30'])
    plt.title(f'TM: {A_matrix_list[0]}')
    plt.xlabel('O Coverage (ML)')
    plt.ylabel('Ag Coverage (ML)')
    plt.savefig('asop.png')
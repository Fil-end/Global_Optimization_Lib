from ase.io import write
from ase.io.dmol import read_dmol_arc, write_dmol_arc
import matplotlib.pyplot as plt
import numpy as np
import yaml

def load_opt_traj(structures:np.asarray, orig_arc:str) -> None:
    atoms = read_dmol_arc(orig_arc)
    atoms_list = []

    for pos in structures:
        slab = atoms.copy()
        slab.positions = pos
        atoms_list.append(slab)

    write('go_traj.xyz', atoms_list)
    write_dmol_arc('go_traj.arc', atoms_list)

def paint_energy_profile(energy:np.asarray) -> None:
    energy = energy - energy[0]
    
    plt.xticks(fontsize=12, fontfamily='Arial', fontweight='bold')
    plt.yticks(fontsize=12, fontfamily='Arial', fontweight='bold')
    plt.title('Energy Profile', fontsize=24, fontweight='bold', fontfamily='Arial')
    plt.xlabel('Epoch', fontsize=18, fontweight='bold', fontstyle='italic', fontfamily='Arial')
    plt.ylabel('Energy(/eV)', fontsize=18, fontweight='bold', fontstyle='italic', fontfamily='Arial')

    plt.plot(energy, color='blue', label = 'HS')
    plt.plot(energy, 'x')

    plt.savefig('./energy_profile.png', bbox_inches='tight')

if __name__ == '__main__':
    with open('./setting.yml', 'r', encoding='utf-8') as f:
        setting = yaml.load(f.read(), Loader=yaml.FullLoader)

    opt_alg = setting['Global-opt']['algorithms']
    npz_file = f"{opt_alg}.npz"
    orig_arc = f"opt_{opt_alg}.arc"

    data = np.load(npz_file, allow_pickle = True)
    structures = data['structures']
    # energy = data['energy']

    energy = data['initial_energy'] + data['global_best_fit'] / 100

    load_opt_traj(structures, orig_arc)
    paint_energy_profile(energy)
from surface_env import MCTEnv
from ase.calculators.lasp_PdO import LASP
from ase.io.lasp_PdO import write_arc, read_arc

def lasp_ssw_calc(atom, type):
    write_arc([atom])
    if type in ['s', 'short']:
        atom.calc = LASP(task='ssw', pot='PdO', potential='NN D3')
    elif type in ['l', 'long']:
        atom.calc = LASP(task='long-ssw', pot='PdO', potential='NN D3')
    energy = atom.get_potential_energy()
    force = atom.get_forces()
    atom = read_arc('allstr.arc', index = -1)
    return atom, energy, force

if __name__ == '__main__':
    energy_list = []
    O_num_list = [0]
    traj = []

    env = MCTEnv(save_dir = 'save_dir', save_every= 20 ,timesteps = 1000, reaction_H = 0.790, reaction_n = 70, 
             delta_s = -0.414, use_GNN_description = True,model_path = 'PdO', calculator_method = 'lasp') 

    initial_slab = env._generate_initial_slab()
    slab, initial_energy, _ = env.calculator(initial_slab)
    initial_energy = initial_energy + env.E_O2 * env.n_O2 + env.E_O3 * env.n_O3
    energy_list.append(initial_energy)
    traj.append(slab.copy())

    for _ in range(70):
        slab = env.choose_ads_site(slab)
        env.to_constraint(slab)
        slab, energy, _ = lasp_ssw_calc(slab, 's')
        energy += env.E_O2 * env.n_O2 + env.E_O3 * env.n_O3
        energy_list.append(energy)
        traj.append(slab.copy())

    best_atoms = traj[energy_list.index(min(energy_list))]
    env.to_constraint(best_atoms)
    best_atoms, best_energy,_ = lasp_ssw_calc(best_atoms, 'l')
    best_energy += env.E_O2 * env.n_O2 + env.E_O3 * env.n_O3
    print(f'energy_list is: {energy_list}')
    print(f'best_energy is: {best_energy}')

    write_arc([best_atoms],  name = 'best_1.arc')
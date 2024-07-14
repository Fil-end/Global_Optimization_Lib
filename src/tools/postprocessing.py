from ase.io.lasp_PdO import read_arc, write_arc
from calc import Calculator
from surface_env import MCTEnv


if __name__ == '__main__':
    # initialize environment and calculator
    env = MCTEnv(calculator_method = 'mace', 
                 model_path = 'PdSiOH.model')
    calculator = Calculator(calculate_method = 'mace', 
                            model_path = 'PdSiOH.model')
    E_O2 = env.E_O2
    E_O3 = env.E_O3

    # calc initial atoms' energy
    initial_atoms = env._generate_initial_slab()
    initial_energy = calculator(initial_atoms, calc_type="single")

    # calc the best atoms generated from GA
    best_atoms = read_arc('best_1.arc')[0]
    best_atoms, del_list = env.del_env_adsorbate(best_atoms)
    if del_list:
        best_atoms, best_energy, _ = calculator(best_atoms)

    write_arc([best_atoms])

    # calc the num of O2 and O3
    if (len(best_atoms) - len(initial_atoms)) % 2:
        n_O3 = 1
        n_O2 = int((len(best_atoms) - len(initial_atoms) - 3) / 2)
    else:
        n_O3 = 0
        n_O2 = int((len(best_atoms) - len(initial_atoms)) / 2)
    
    energy = best_energy - (initial_energy + n_O2 * E_O2 + n_O3 * E_O3)
    print(energy / len(best_atoms))
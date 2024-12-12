from ast import literal_eval
from dataclasses import dataclass
import random
from typing import List, Optional
from timeout_decorator import timeout

from ase import Atoms
from ase.constraints import FixAtoms
from ase.ga.utilities import (closest_distances_generator,
                              atoms_too_close,
                              get_all_atom_types)
from ase.io import read, write
from ase.io.dmol import read_dmol_arc, write_dmol_arc
import numpy as np
import yaml

from EDRL.tools.calc import Calculator
#from surface_env import MCTEnv

#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Target system: PdO, PdC, CuO, CuZnO, MoC
# PdCH, PdOH, etc
# For direct example: PdO, PdC, CuO
# Comparison: GA, PSO, LC-SSW, etc.

REGISTER_GO_LIST = ['GWO', 'IGWO']

@dataclass
class GWO():
    calculator_method: str = 'MACE'
    model_path: str = 'PdSiOH.model'
    max_iterations: int = 20
    pack_size: int = 5
    vector_size: int = 3
    traj_file:str = 'save.xyz'

    def __post_init__(self) -> None:
        self.calculator = Calculator(calculate_method = self.calculator_method, 
                                     model_path = self.model_path)

    # generate wolf
    # In atoms simulation environment
    def wolf(self, 
             vector_size: int, 
             min_range: np.asarray, 
             max_range: np.asarray,
             ele_list:List) -> np.asarray:
        wolf_position = [np.array([[0.0, 0.0, 0.0] for _ in range(len(ele_list))]) \
                         for _ in range(vector_size)]
    
        for i in range(vector_size):
            wolf_position[i] = ((max_range - min_range) * np.random.rand(len(ele_list),3).astype(float) + min_range)
        return wolf_position
  
    # generate wolf pack
    def pack(self, ele_list:List, box:List) -> List[np.asarray]:
        pack = [self.wolf(self.vector_size, 
                          np.array([box[0] for _ in range(len(ele_list))]), 
                          np.array([box[0] + [box[1][0][0], box[1][1][1], box[1][2][2]] \
                                    for _ in range(len(ele_list))]), ele_list) \
                          for _ in range(self.pack_size)]
        return pack
    
    def fitness(self,
                org_atoms:Atoms,
                individual:np.asarray, 
                ele_list:Optional[List]=None, 
                blmin:np.asarray = None) -> float:
        # fitness_list = []
        fitness = 0
        for pos in individual:
            add_atoms = Atoms(positions = np.array(pos), 
                              numbers = ele_list, 
                              cell = org_atoms.get_cell())
            atoms = org_atoms + add_atoms
            if not atoms_too_close(atoms, blmin):
                try:
                    _, energy = self.get_energy(atoms)
                    fitness += energy
                except:
                    fitness += 0
        return fitness
    
    def hunt(self, 
             org_atoms:Optional[Atoms] = None,
             ele_list:Optional[List] = None,
             box:Optional[np.asarray] = None) -> None:
        best_wolf = []
        # generate min pair dirstance
        unique_atom_types = get_all_atom_types(org_atoms, ele_list)
        blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                            ratio_of_covalent_radii=0.25)
        # generate wolf pack
        wolf_pack = self.pack(ele_list, box)
        # sort pack by fitness
        pack_fit = sorted([(self.fitness(org_atoms, i, ele_list, blmin), i) for i in wolf_pack])
  
        # main loop
        for k in range(self.max_iterations):
            #choose best 3 solutions
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            
            print(f'iteration: {k}, best_wolf_position: {self.fitness(org_atoms,alpha, ele_list, blmin)}')
            
            # linearly decreased from 2 to 0
            a = 2*(1 - k/self.max_iterations)
            
            # updating each population member with the help of alpha, beta and delta
            for i in range(self.pack_size):
                # compute A and C 
                A1, A2, A3 = a * (2 * np.random.random(3) - 1)
                C1, C2, C3 = 2 * np.random.random(3)

                # generate vectors for new position
                X1 = [np.array([[0.0, 0.0, 0.0] for _ in range(len(ele_list))]) \
                                for _ in range(self.vector_size)]
                X2 = [np.array([[0.0, 0.0, 0.0] for _ in range(len(ele_list))]) \
                                for _ in range(self.vector_size)]
                X3 = [np.array([[0.0, 0.0, 0.0] for _ in range(len(ele_list))]) \
                                for _ in range(self.vector_size)]
                new_position = [np.array([[0.0, 0.0, 0.0] for _ in range(len(ele_list))]) \
                                for _ in range(self.vector_size)]
                
                # hunting 
                for j in range(self.vector_size):
                    X1[j] = alpha[j] - A1 * abs(C1 - alpha[j] - wolf_pack[i][j])
                    X2[j] = beta[j] - A2 * abs(C2 - beta[j] - wolf_pack[i][j])
                    X3[j] = delta[j] - A3 * abs(C3 - delta[j] - wolf_pack[i][j])
                    new_position[j] += (X1[j] + X2[j] + X3[j]) / 3
                
                # fitness calculation of new position
                new_fitness = self.fitness(org_atoms,new_position, ele_list, blmin)
                
                # if new position is better then replace, greedy update
                if new_fitness < self.fitness(org_atoms,wolf_pack[i], ele_list, blmin):
                    wolf_pack[i] = new_position

            # sort the new positions by their fitness
            pack_fit = sorted([(self.fitness(org_atoms,i, ele_list, blmin), i) for i in wolf_pack])
            best_wolf.append(pack_fit[0][1])

        self.save_traj(org_atoms, ele_list, best_wolf)
    
    @timeout(30)
    def get_energy(self, atoms:Atoms) -> float:
        atoms.pbc = True
        write_dmol_arc('input.arc', [atoms])
        atoms, energy, force =self.calculator(atoms)
        atoms.pbc = True
        write_dmol_arc('chkpt.arc', [atoms])
        if np.max(force) > 5:
            return atoms, 0
        else:
            return atoms, energy
    
    def save_traj(self, org_atoms:Optional[Atoms], ele_list, best_wolf:List) -> None:
        traj = []
        for individual in best_wolf:
            for pos in individual:
                add_atoms = Atoms(positions = np.array(pos), 
                                  numbers = ele_list, 
                                  cell = org_atoms.get_cell())
                atoms = org_atoms + add_atoms
                atoms, _ = self.get_energy(atoms)
                traj.append(atoms)

        write(self.traj_file, traj)
        
def main(setting, 
         org_atoms: Optional[Atoms] = None, 
         atom_numbers: Optional[List] = None, 
         box: Optional[np.asarray] = None,):
    
    if org_atoms is not None and box is None:
        box = org_atoms.cell
    
    opt_alg = setting['Global-opt']['algorithms']
    if opt_alg in REGISTER_GO_LIST:
        model = eval(opt_alg)(calculator_method = setting['Calculator'],
                            model_path = setting['model'],
                            max_iterations = int(setting['Global-opt']['max_interation']), 
                            pack_size = int(setting['Global-opt']['pack_size']), 
                            vector_size = int(setting['Global-opt']['vector_size']),
                                )
        
        model.hunt(org_atoms, atom_numbers, box)
    else:
        raise ValueError(f"Your current global optimization algorithm is {opt_alg}, \
            global optimization algorithm should in {REGISTER_GO_LIST}!!!")

if __name__ == '__main__':
    with open('./setting.yml', 'r', encoding='utf-8') as f:
        setting = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    if 'atoms_path' in setting['org_atoms']:
        file_path = setting['org_atoms']['atoms_path']
        file_type = file_path.split('.')[-1]
        if file_type in ['arc']:
            slab = read_dmol_arc(file_path)
        else:
            slab = read(file_path)

    elif setting['org_atoms']['Geometry'] in ['Surface', 'surface']:
        from ase.build import (fcc111, fcc110, fcc100, 
                                bcc100, hcp10m10, diamond100)
        facet = setting['org_atoms']['facet']
        if facet in ['fcc111', 'fcc110', 'fcc100', 
                      'bcc100', 'hcp10m10', 'diamond100']:
            size = literal_eval(setting['org_atoms']['size'])
            metal = setting['metal']
            slab = eval(facet)(metal, size = size, vacuum = 10.0)
            pos = slab.get_positions()
            cell = slab.get_cell()
            p0 = np.array([0., 0., max(pos[:, 2])])
            v1 = cell[0, :]
            v2 = cell[1, :]
            v3 = cell[2, :]
            v3[2] = 3.5
        else:
            raise ValueError(f"Your current facet is {facet}, facet should in \
                ('fcc111', 'fcc110', 'fcc100', 'bcc100', 'hcp10m10', 'diamond100')!!!")

    elif setting['org_atoms']['Geometry'] in ['Cluster', 'cluster']:
        slab = Atoms()
        cell = literal_eval(setting['org_atoms']['box'])
        p0 = np.array([0., 0., 0.])
        v1 = np.array([cell[0], 0., 0.])
        v2 = np.array([0., cell[1], 0.])
        v3 = np.array([0., 0., cell[2]])

    atom_numbers = eval(setting['Opt_atoms'])

    slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

    main(org_atoms = slab,
         atom_numbers = atom_numbers,
         box = [p0, [v1, v2, v3]],
         setting = setting)

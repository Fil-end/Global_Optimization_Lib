from dataclasses import dataclass
from typing import List, Optional
from timeout_decorator import timeout

from ase import Atoms
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (closest_distances_generator,
                              atoms_too_close,
                              get_all_atom_types)
from ase.io import  write
from ase.io.dmol import  write_dmol_arc
import numpy as np

from EDRL.tools.calc import Calculator
#from surface_env import MCTEnv

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
    def pack(self, sg:StartGenerator) -> List[np.asarray]:
        pack = []
        for _ in range(self.pack_size):
            wolfs = []
            for _ in range(self.vector_size):
                new_state = sg.get_new_candidate(maxiter = 10000)
                if new_state is not None:
                    wolfs.append(new_state)
            pack.append(wolfs)
        return pack
    
    def fitness(self, individual:List[Atoms]) -> float:
        fitness = 0
        for atoms in individual:
            atoms.pbc = True
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
                                            ratio_of_covalent_radii=0.6)
        
        sg = StartGenerator(org_atoms, ele_list, blmin,
                            box_to_place_in=box)
        # generate wolf pack
        wolf_pack = self.pack(sg)
        # sort pack by fitness
        pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
  
        # main loop
        for k in range(self.max_iterations):
            #choose best 3 solutions
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            print(f'iteration: {k}, best_wolf_position: {self.fitness(alpha)}')
            
            # linearly decreased from 2 to 0
            a = 2*(1 - k/self.max_iterations)
            
            # updating each population member with the help of alpha, beta and delta
            for i in range(self.pack_size):
                # compute A and C 
                A1, A2, A3 = a * (2 * np.random.random(3) - 1)
                C1, C2, C3 = 2 * np.random.random(3)

                # generate vectors for new position
                X1 = [np.array([[0.0, 0.0, 0.0] for _ in range(len(alpha[0]))]) \
                                for _ in range(self.vector_size)]
                X2 = [np.array([[0.0, 0.0, 0.0] for _ in range(len(alpha[0]))]) \
                                for _ in range(self.vector_size)]
                X3 = [np.array([[0.0, 0.0, 0.0] for _ in range(len(alpha[0]))]) \
                                for _ in range(self.vector_size)]
                new_position = [np.array([[0.0, 0.0, 0.0] for _ in range(len(alpha[0]))]) \
                                for _ in range(self.vector_size)]
                
                # hunting 
                for j in range(self.vector_size):
                    X1[j] = alpha[j].get_positions() - A1 * abs(C1 - alpha[j].get_positions() - wolf_pack[i][j].get_positions())
                    X2[j] = beta[j].get_positions() - A2 * abs(C2 - beta[j].get_positions() - wolf_pack[i][j].get_positions())
                    X3[j] = delta[j].get_positions() - A3 * abs(C3 - delta[j].get_positions() - wolf_pack[i][j].get_positions())
                    new_position[j] += (X1[j] + X2[j] + X3[j]) / 3
                    
                new_candidate = []
                for pos in new_position:
                    state = alpha[0].copy()
                    state.set_positions(pos)
                    new_candidate.append(state)
                
                # fitness calculation of new position
                new_fitness = self.fitness(new_candidate)
                
                # if new position is better then replace, greedy update
                if new_fitness < self.fitness(wolf_pack[i]):
                    wolf_pack[i] = new_candidate

            # sort the new positions by their fitness
            pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
            best_wolf.append(pack_fit[0][1])

        self.save_traj(best_wolf)
    
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
    
    def save_traj(self,  best_wolf:List[Atoms]) -> None:
        for atoms in best_wolf:
            atoms.pbc = True
        write(self.traj_file, best_wolf)
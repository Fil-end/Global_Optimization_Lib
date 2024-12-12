from dataclasses import dataclass
import random
from typing import List, Optional
from timeout_decorator import timeout

from ase import Atoms
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (closest_distances_generator,
                              get_all_atom_types)
from ase.io import write
from ase.io.dmol import write_dmol_arc
import numpy as np

from EDRL.tools.calc import Calculator


@dataclass
class IGWO():
    calculator_method: str = 'MACE'
    model_path: str = 'PdSiOH.model'
    max_iterations: int = 20
    pack_size: int = 5
    vector_size: int = 3
    traj_file:str = 'save_IGWO.xyz'

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
        fitness = []
        for atoms in individual:
            atoms.pbc = True
            try:
                _, energy = self.get_energy(atoms)
                fitness.append(energy)
            except:
                fitness.append(0)
        return min(fitness)
    
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
                
                X_GWO = []
                for pos in new_position:
                    state = alpha[0].copy()
                    state.set_positions(pos)
                    X_GWO.append(state)
                # fitness calculation of new position
                new_fitness = self.fitness(X_GWO)

                # current wolf fitness
                current_wolf = wolf_pack[i]

                # Begin i-GWO ehancement, Compute R --------------------------------
                current_fitness = self.fitness(current_wolf)
                R = current_fitness - new_fitness
                
                # Compute eq. 11, build the neighborhood
                neighborhood = []
                for l in wolf_pack:
                    neighbor_distance = current_fitness - self.fitness(l)
                    if neighbor_distance <= R:
                        neighborhood.append(l)

                # if the neigborhood is empy, compute the distance with respect 
                # to the other wolfs in the population and choose the one closer
                closer_neighbors = []
                if len(neighborhood) == 0:
                    for n in wolf_pack:
                        distance_wolf_alone = current_fitness - self.fitness(n)
                        closer_neighbors.append((distance_wolf_alone,n))
                        
                    closer_neighbors = sorted(closer_neighbors)
                    neighborhood.append(closer_neighbors[0][1])

                # Compute eq. 12 compute new candidate using neighborhood
                DLH_positions = [np.array([[0.0, 0.0, 0.0] for _ in range(len(alpha[0]))]) \
                                for _ in range(self.vector_size)]
                for m in range(self.vector_size):
                    random_neighbor = random.choice(neighborhood)
                    random_wolf_pop = random.choice(wolf_pack)
                    
                    DLH_positions[m] = current_wolf[m].get_positions() + \
                        random.random() * random_neighbor[m].get_positions() - random_wolf_pop[m].get_positions()
                
                X_DLH = []
                for pos in new_position:
                    state = alpha[0].copy()
                    state.set_positions(pos)
                    X_DLH.append(state)
                # if X_GWO is better than X_DLH, select X_DLH
                DLH_fitness = self.fitness(X_DLH)
                if new_fitness  < DLH_fitness:
                    candidate = X_GWO
                else:
                    candidate = X_DLH
                    new_fitness = DLH_fitness

                # if new position is better then replace, greedy update
                if new_fitness < self.fitness(wolf_pack[i]):
                    wolf_pack[i] = candidate

            # sort the new positions by their fitness
            pack_fit = sorted([(self.fitness(i), i) for i in wolf_pack])
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
                atoms.pbc = True
                traj.append(atoms)

        write(self.traj_file, traj)
        write_dmol_arc('save_igwo.arc', traj)
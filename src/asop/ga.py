from dataclasses import dataclass
import logging
from random import random
from typing import List

from ase import Atoms
from ase.build import surface
from ase.constraints import FixAtoms
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.population import Population
from ase.ga.offspring_creator import OperationSelector
from ase.ga.standardmutations import (MirrorMutation,
                                      RattleMutation,
                                      PermutationMutation)
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (closest_distances_generator,
                              get_all_atom_types)
from ase.io import write, Trajectory
import numpy as np

from calc import Calculator
from surface_env import MCTEnv


logger = logging.getLogger(__name__)

@dataclass
class GA():
    calculator_method: str = 'MACE'
    model_path: str = 'PdSiOH.model'
    traj_path: str = 'PdO_test_101_candidates.traj'
    db_file: str = 'PdO_test_101_gadb.db'
    # Change the following three parameters to suit your needs
    population_size:int = 20
    mutation_probability:float = 0.3
    n_to_test:int = 200

    def __post_init__(self) -> None:
        self.env = MCTEnv(calculator_method = self.calculator_method, 
                                     model_path = self.model_path)
        self.calculator = Calculator(calculate_method = self.calculator_method, 
                                     model_path = self.model_path)
        
    def __call__(self, slab:Atoms = None, atom_numbers:List = None) -> None:
        self.population(slab, atom_numbers)
        self.propoganda()
        return self.postprocessing()
    
    def population(self, slab:Atoms, atom_numbers:List=None):
        # create the surface
        if slab is None:
            slab = surface('Pd',(1,0,1),3)
            cell = slab.cell
            cell[2][2] = 30.0
            slab.set_cell(cell)
            for atom in slab:
                atom.position[2] += 10.0
            slab = slab * (2,2,1)
        slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

        # define the volume in which the adsorbed cluster is optimized
        # the volume is defined by a corner position (p0)
        # and three spanning vectors (v1, v2, v3)
        pos = slab.get_positions()
        cell = slab.get_cell()
        p0 = np.array([0., 0., max(pos[:, 2]) + 2.])
        v1 = cell[0, :] * 0.8
        v2 = cell[1, :] * 0.8
        v3 = cell[2, :]
        v3[2] = 3.

        # Define the composition of the atoms to optimize
        if atom_numbers is None:
            atom_numbers = 16 * [46] + 32 * [8]

        # define the closest distance two atoms of a given species can be to each other
        unique_atom_types = get_all_atom_types(slab, atom_numbers)
        blmin = closest_distances_generator(atom_numbers=unique_atom_types,
                                            ratio_of_covalent_radii=0.7)

        # create the starting population
        sg = StartGenerator(slab, atom_numbers, blmin,
                            box_to_place_in=[p0, [v1, v2, v3]])

        # generate the starting population
        starting_population = []
        for _ in range(self.population_size):
            new_state = sg.get_new_candidate(maxiter = 1000)
            if new_state is not None:
                starting_population.append(new_state)

        # create the database to store information in
        d = PrepareDB(db_file_name=self.db_file,
                      simulation_cell=slab,
                      stoichiometry=atom_numbers)

        for a in starting_population:
            d.add_unrelaxed_candidate(a)
    
    def propoganda(self):
        # Initialize the different components of the GA
        da = DataConnection(self.db_file)
        atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
        n_to_optimize = len(atom_numbers_to_optimize)
        slab = da.get_slab()
        all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
        blmin = closest_distances_generator(all_atom_types,
                                            ratio_of_covalent_radii=0.7)

        comp = InteratomicDistanceComparator(n_top=n_to_optimize,
                                            pair_cor_cum_diff=0.015,
                                            pair_cor_max=0.7,
                                            dE=0.02,
                                            mic=False)

        pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
        mutations = OperationSelector([1., 1., 1.],
                                    [MirrorMutation(blmin, n_to_optimize),
                                    RattleMutation(blmin, n_to_optimize),
                                    PermutationMutation(n_to_optimize)])

        # Relax all unrelaxed structures (e.g. the starting population)
        while da.get_number_of_unrelaxed_candidates() > 0:
            a = da.get_an_unrelaxed_candidate()
            info = a.info
            self.env.to_constraint(a)
            a, _, _ = self.calculator(a)
            self.get_ga_info(a, info)
            logger.info('Relaxing starting candidate {0}'.format(a.info['confid']))
            a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
            da.add_relaxed_step(a)

        # create the population
        population = Population(data_connection=da,
                                population_size=self.population_size,
                                comparator=comp)

        # test n_to_test new candidates
        for i in range(self.n_to_test):
            logger.info('Now starting configuration number {0}'.format(i))
            a1, a2 = population.get_two_candidates()
            a1.set_pbc([True, True, False])
            a2.set_pbc([True, True, False])
            a3, desc = pairing.get_new_individual([a1, a2])
            if a3 is None:
                continue
            da.add_unrelaxed_candidate(a3, description=desc)

            # Check if we want to do a mutation
            if random() < self.mutation_probability:
                a3_mut, desc = mutations.get_new_individual([a3])
                if a3_mut is not None:
                    da.add_unrelaxed_step(a3_mut, desc)
                    a3 = a3_mut

            # Relax the new candidate
            info = a3.info
            self.env.to_constraint(a3)
            a3, _, _ = self.calculator(a3)
            self.get_ga_info(a3, info)
            a3.info['key_value_pairs']['raw_score'] = -a3.get_potential_energy()
            da.add_relaxed_step(a3)
            population.update()

        write(self.traj_path, da.get_all_relaxed_candidates())

    def postprocessing(self) -> Atoms:
        traj = Trajectory(self.traj_path, 'r')
        energy_list = []
        for atoms in traj:
            energy_list.append(atoms.get_potential_energy())
        return traj[0]

    def get_ga_info(self,atoms:Atoms, info) -> None:
        atoms.info = info

    @property
    def initial_slab(self) -> Atoms:
        return self.env._generate_initial_slab()
    
    @property
    def initial_energy(self):
        energy = self.calculator(self.initial_slab, calc_type="single")
        return energy + 24 * self.env.E_O2
    
    def energy_output(self):
        traj = Trajectory(self.traj_path)
        energy_list = []
        for atoms in traj:
            energy_list.append(atoms.get_potential_energy() - self.initial_energy)
        return energy_list

if __name__ == '__main__':
    ga = GA()
    energy_output = ga()
    print(energy_output)
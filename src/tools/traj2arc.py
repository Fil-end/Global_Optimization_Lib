from ase.io import Trajectory
from ase.io.lasp_PdO import write_single, add_text_head
import os
import numpy as np

traj = Trajectory('PdO_island_all_candidates.traj', 'r')
energy_list = []
for atoms in traj:
    energy_list.append(atoms.get_potential_energy())

print(energy_list)

write_single(traj[0], file_name = 'best_1.arc', write_type='a')

add_text_head(filename="best_1.arc")

'''write_single(traj[np.random.randint(len(traj))], file_name = 'test_2.arc', write_type='a')

add_text_head(filename="test_2.arc")

write_single(traj[-1], file_name = 'test_3.arc', write_type='a')

add_text_head(filename="test_3.arc")'''



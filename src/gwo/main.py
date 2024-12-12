from ast import literal_eval
from typing import List, Optional

from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.dmol import read_dmol_arc
import numpy as np
import yaml

from GWO import GWO
from IGWO import IGWO


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
            v3[2] = 3.
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

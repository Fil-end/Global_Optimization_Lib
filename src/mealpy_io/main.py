from ast import literal_eval
from dataclasses import dataclass
import os
from typing import List, Optional, Tuple, Dict

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.data import chemical_symbols
from ase.io import read
from ase.io.dmol import read_dmol_arc, write_dmol_arc
from ase.optimize import LBFGS
import numpy as np
from spglib import get_symmetry_dataset
from timeout_decorator import timeout

import mealpy
from mealpy import *
import yaml

from calc import Calculator
from EDRL.tools.periodic_table import METALLIST, NONMETALLIST
from EDRL.tools.sites import Sites

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

OPT_ALGORITHMS_FACTORY = mealpy.get_all_optimizers()

@dataclass
class EquivariantAtomsProblem(Problem):
    orig_atoms: Atoms
    ele_list: List
    setting: Dict
    box: np.asarray = None

    # Currently not used
    symprec: float = 0.2
    delta: float = 1.0
    use_symmetry: bool = True

    def __post_init__(self, **kwargs):
        self.calculate_method = self.setting['Calculator']
        self.model_path = self.setting['model']
        self.opt_alg = self. setting['Global-opt']['algorithms']

        self._order_ele_list()

        if os.path.exists(f'opt_{self.opt_alg}.arc'):
            self.atoms = read_dmol_arc(f'opt_{self.opt_alg}.arc')
            self.atoms.set_constraint(FixAtoms(mask=len(self.orig_atoms) * [True]))
        else:
            self.atoms = Sites(self.orig_atoms, self.ele_list)()
            self.atoms.pbc = True
            write_dmol_arc('input.arc', [self.atoms])
            raise ValueError('stop')

        self.atoms.pbc = True
        self.atoms, self.initial_energy = self._opt(self.atoms)

        # 定义优化变量边界（原子坐标）
        max_free_z = max(self.orig_atoms.positions[:, 2]) + 0.5
        pos = self.atoms.positions[-len(self.ele_list):].flatten()

        bounds = []
        for p_idx in range(len(pos)):
            if (p_idx + 1) % 3 == 0:
                bounds.append((max(max_free_z, pos[p_idx]-2.0), 
                               min(max_free_z + self.box[1][2][2], pos[p_idx]+2.0)))
            elif (p_idx + 1) % 3 == 1:
                bounds.append((0, cell[0][0]))
            elif (p_idx + 1) % 3 == 2:
                bounds.append((0, cell[1][1]))

        # 初始化mealpy问题
        super().__init__(
            bounds=[FloatVar(lb, ub) for lb, ub in bounds],
            minmax="min",
            obj_func=self._calculate_energy,
            **kwargs
        )

    def _order_ele_list(self) -> None:
        '''In self.ele_list, metal element should be former than nonmetal element'''
        metal_list = []
        nonmetal_list = []

        try:
            for ele_idx in self.ele_list:
                if chemical_symbols[ele_idx] in METALLIST:
                    metal_list.append(ele_idx)
                elif chemical_symbols[ele_idx] in NONMETALLIST:
                    nonmetal_list.append(ele_idx)
        except:
            raise ValueError('Your input element index should in (1-103)')
        
        metal_list.extend(nonmetal_list)
        self.ele_list = metal_list

    def _get_symmetry_operations(self, atoms: Atoms) -> Optional[dict]:
        """修复：添加spglib的异常处理"""
        try:
            slab = atoms.copy()
            dataset = get_symmetry_dataset(
                (slab.cell, slab.positions, slab.get_atomic_numbers()), symprec=self.symprec)

            return {
                "rotations": dataset.rotations,
                "translations": dataset.translations,
                "equivalent_atoms": dataset.equivalent_atoms
            }
        except Exception as e:
            print(f"对称性分析失败: {str(e)}")
            return None

    def _reconstruct_structure(self, solution):
        # TODO: check whether the target solution is incorrect
        atoms = self.atoms.copy()
        atoms.positions[-len(self.ele_list):] = solution.reshape(-1, 3)
        atoms, energy = self._opt(atoms)
        return atoms, energy

    def _calculate_energy(self, solution):
        """修复：添加详细的错误处理"""
        try:
            _, energy = self._reconstruct_structure(solution)
            return 100 * (energy - self.initial_energy)

        except Exception as e:
            raise ValueError(f"能量计算失败: {str(e)}", exc_info=True)

    @timeout(30)
    def _opt(self, atoms: Atoms) -> Tuple[Atoms, float]:
        try:
            if self.calculate_method == 'EMT':
                atoms.calc = EMT()
                dyn = LBFGS(atoms, trajectory='lbfgs.traj')
                dyn.run(steps = 200, fmax = 0.1)
                energy = atoms.get_potential_energy()
                force = atoms.get_forces()
            else:
                calculator = Calculator(calculate_method=self.calculate_method,
                                        model_path=self.model_path)
                atoms, energy, force = calculator(atoms)
            
            if np.max(force) > 5:
                return atoms, 0
            else:
                return atoms, energy
        except:
            return atoms, 0
        
    def save_episode(self, history: Dict) -> None:
        atoms = self.atoms.copy()
        global_best_structure = []
        global_best_energy = []

        for target in history.list_global_best:
            atoms.positions[-len(self.ele_list):] = target.solution.reshape(-1, 3)
            atoms, energy = self._opt(atoms)
            global_best_structure.append(atoms.positions)
            global_best_energy.append(energy)

        save_path = os.path.join('.', f'{self.opt_alg}.npz')
        np.savez_compressed(
                save_path,
                
                initial_energy=self.initial_energy,
                structures =  global_best_structure,
                energy = global_best_energy,
                global_best = history.list_global_best,
                global_best_fit = history.list_global_best_fit,
                epoch_time = history.list_epoch_time,
                diversity = history.list_diversity,
                exploitation = history.list_exploitation,
                exploration = history.list_exploration,
                population = history.list_population,
            )

def main(setting,
         orig_atoms: Optional[Atoms] = None,
         atom_numbers: Optional[List] = None,
         box: Optional[np.asarray] = None,
         **kwargs):

    if orig_atoms is not None and box is None:
        box = orig_atoms.cell

    opt_alg = setting['Global-opt']['algorithms']

    if opt_alg in OPT_ALGORITHMS_FACTORY:
        problem = EquivariantAtomsProblem(orig_atoms=orig_atoms,
                                          ele_list=atom_numbers,
                                          box=box,
                                          setting = setting,
                                          **kwargs)
        
        model = mealpy.get_optimizer_by_name(opt_alg) \
                            (int(setting['Global-opt']['epoch']), int(setting['Global-opt']['pop_size']))

        best_solution = model.solve(problem,
                                    atoms = problem.atoms,
                                    local_opt = True,
                                    calculate_method=setting['Calculator'],
                                    model_path=setting['model'])
        
        opt_slab, _ = problem._reconstruct_structure(best_solution.solution)

        opt_slab.pbc = True
        write_dmol_arc(f"opt_{setting['Global-opt']['algorithms']}.arc", [opt_slab])

        problem.save_episode(model.history)
    else:
        raise ValueError(f"{opt_alg} not in Mealpy algorithms !")

    return opt_slab

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
            slab = eval(facet)(metal, size=size, vacuum=10.0)
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

    opt_slab = main(orig_atoms=slab,
                    atom_numbers=atom_numbers,
                    box=[p0, [v1, v2, v3]],
                    setting=setting)

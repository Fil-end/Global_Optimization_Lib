from ast import literal_eval
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.data import chemical_symbols
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (closest_distances_generator,
                              get_all_atom_types)
from ase.io import read
from ase.io.dmol import read_dmol_arc, write_dmol_arc
from ase.optimize import LBFGS
import logging
import numpy as np
from spglib import get_symmetry_dataset
from timeout_decorator import timeout

from mealpy import *
import yaml

from calc import Calculator
from EDRL.tools.periodic_table import METALLIST, NONMETALLIST
from EDRL.tools.sites import Sites

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

OPT_ALGORITHMS_FACTORY = {
    "GWO": GWO.OriginalGWO,
    "GWO_WOA": GWO.GWO_WOA,
    "PSO": PSO.OriginalPSO,
    "WOA": WOA.OriginalWOA,
    "HI_WOA": WOA.HI_WOA,
}

@dataclass
class EquivariantAtomsProblem(Problem):
    orig_atoms: Atoms
    ele_list: List
    calculate_method: str
    model_path: str = None
    box: np.asarray = None

    # Currently not used
    symprec: float = 0.2
    delta: float = 1.0
    use_symmetry: bool = True

    def __post_init__(self, **kwargs):
        # unique_atom_types = get_all_atom_types(self.orig_atoms, self.ele_list)
        # blmin = closest_distances_generator(atom_numbers=unique_atom_types,
        #                                     ratio_of_covalent_radii=0.6)
        
        # self.sg = StartGenerator(self.orig_atoms, self.ele_list, blmin,
        #                     box_to_place_in=self.box)

        # self.atoms = self.sg.get_new_candidate(maxiter = 10000)
        self._order_ele_list()
        self.atoms = Sites(self.orig_atoms, self.ele_list)()
        self.atoms.pbc = True

        # 对称性分析
        # self.sym_ops = self._get_symmetry_operations(atoms) if self.use_symmetry else None

        # 生成优化变量参数化
        # TODO: 将orig_atoms 作为模版基底，将 ele_list 作为待优化原子的元素列表
        # 并将box作为限制条件
        # self.param_mapping = self._parameterize_variables(atoms)
        # print(self.param_mapping)

        # 定义优化变量边界（原子坐标）
        max_free_z = max(self.orig_atoms.positions[:, 2]) + 0.5
        pos = self.atoms.positions[-len(self.ele_list):].flatten()

        bounds = []
        for p_idx in range(len(pos)):
            if (p_idx + 1) % 3 == 0:
                bounds.append((max(max_free_z, pos[p_idx]-2.0), 
                               min(max_free_z + self.box[1][2][2], pos[p_idx]+2.0)))
            else:
                bounds.append((pos[p_idx]-5.0, pos[p_idx]+5.0))

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

    '''def _parameterize_variables(self, atoms:Atoms) -> Dict:
        """修复：对称性参数化的维度问题"""
        pos = atoms.positions
        n_atoms = len(pos)

        # 检测自由原子
        mask = np.zeros((n_atoms, 3), bool)
        if self.orig_atoms.constraints:
            for constr in self.orig_atoms.constraints:
                if isinstance(constr, FixAtoms):
                    mask[constr.index] = True
        free_mask = ~mask.any(axis=1)
        
        # 对称性参数化
        if self.use_symmetry and self.sym_ops:
            equiv_groups = self._group_equivalent_atoms()
            variables = []
            bounds = []

            # 收集独立变量
            processed = set()
            for group in equiv_groups:
                rep = min(group)
                if free_mask[rep] and rep not in processed:
                    processed.add(rep)
                    variables.extend(pos[rep])
                    bounds.extend([(x-self.delta, x+self.delta) for x in pos[rep]])

            return {
                "vars": np.array(variables),
                "bounds": bounds,
                "mask": free_mask,
                "equiv_groups": equiv_groups
            }
        else:
            # 常规参数化
            flat_pos = pos[free_mask].flatten()
            return {
                "vars": flat_pos,
                "bounds": [(x-self.delta, x+self.delta) for x in flat_pos],
                "mask": free_mask,
                "equiv_groups": None
            }'''

    def _group_equivalent_atoms(self):
        """修复：处理空对称操作的情况"""
        if not self.sym_ops:
            return []
        equiv_atoms = self.sym_ops["equivalent_atoms"]
        groups = {}
        for i, eq in enumerate(equiv_atoms):
            groups.setdefault(eq, []).append(i)
        return list(groups.values())

    def _reconstruct_structure(self, solution):
        """修复：结构重建中的计算器设置和对称性应用"""
        '''atoms = self.orig_atoms.copy()
        atoms.calc = self.orig_atoms.calc  # 确保计算器存在
        solution = np.array(solution, dtype=float)

        if self.param_mapping["equiv_groups"] and self.sym_ops:
            # 对称性约束重建
            new_pos = atoms.positions.copy()
            var_pos = solution.reshape(-1, 3)
            current_var = 0

            for group in self.param_mapping["equiv_groups"]:
                rep = min(group)
                if self.param_mapping["mask"][rep]:
                    base_pos = var_pos[current_var]
                    for atom in group:
                        if self.param_mapping["mask"][atom]:
                            new_pos[atom] = base_pos
                    current_var += 1
            atoms.positions = new_pos
        else:
            # 常规重建
            free_pos = atoms.positions[self.param_mapping["mask"]]
            free_pos[:] = solution.reshape(free_pos.shape)
            atoms.positions[self.param_mapping["mask"]] = free_pos'''
        
        # TODO: check whether the target solution is incorrect
        atoms = self.atoms.copy()
        atoms.positions[-len(self.ele_list):] = solution.reshape(-1, 3)
        atoms, energy = self._opt(atoms)
        return atoms, energy

    def _calculate_energy(self, solution):
        """修复：添加详细的错误处理"""
        try:
            _, energy = self._reconstruct_structure(solution)
            return energy

        except Exception as e:
            logging.error(f"能量计算失败: {str(e)}", exc_info=True)
            return np.inf

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
            return atoms, np.inf


def main(setting,
         orig_atoms: Optional[Atoms] = None,
         atom_numbers: Optional[List] = None,
         box: Optional[np.asarray] = None,
         **kwargs):

    if orig_atoms is not None and box is None:
        box = orig_atoms.cell

    opt_alg = OPT_ALGORITHMS_FACTORY[setting['Global-opt']['algorithms']]
    # if opt_alg in REGISTER_GO_LIST:
    problem = EquivariantAtomsProblem(orig_atoms=orig_atoms,
                                        ele_list=atom_numbers,
                                        box=box,
                                        calculate_method=setting['Calculator'],
                                        model_path=setting['model'],
                                        **kwargs)

    model = opt_alg(int(setting['Global-opt']['epoch']), 
                    int(setting['Global-opt']['pop_size']))

    best_solution = model.solve(problem)
    opt_slab, _ = problem._reconstruct_structure(best_solution.solution)

    opt_slab.pbc = True
    write_dmol_arc(f"opt_{setting['Global-opt']['algorithms']}.arc", [opt_slab])

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
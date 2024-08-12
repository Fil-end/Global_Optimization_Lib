from typing import List,Tuple
from dataclasses import dataclass
import torch

import ase
from ase import units
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin
from ase.optimize import LBFGS

from EDRL.io.lasp import write_arc, read_arc
from lasp import LASP

LASP_COMMAND = 'unset $(compgen -v | grep SLURM); unset SLURM_PROCID;mpirun -np 24 lasp'
@dataclass
class Calculator:
    model_path: str = None
    calculate_method: str = None
    ts_method:str = ''
    temperature_K: float = 473.15

    def __call__(self, atoms:ase.Atoms, calc_type:str = 'opt'):
        '''The calculator provide io for different kind of Machine 
        Learning calculators. Currently, it supports the io for 'MACE',
        'Nequip' and 'LASP'.

        Args:
            atoms (Atoms): The atoms systems.
            calc_type (str): The calculation task. Currently, here only
            support the calc_type in ['single', 'opt', 'md', 'ssw', 'ts']
                'single': calculate single point energy
                'opt': optimization the structure and return its optimized
                    structure, energy and force.
                'md': Molecular dynamics simulation and return its structure
                'ssw': Only in 'LASP'. Stochastic surface walking. A global
                    optimization method.
                'ts': Transition structure search and return its barrier.
        '''
        if self.calculate_method in ["LASP", "Lasp", "lasp"]:
            if calc_type == "opt":
                atoms, energy, force = self.lasp_calc(atoms)
                return atoms, energy, force
            elif calc_type in ["single-point", "single"]:
                energy = self.lasp_single_calc(atoms)
                return energy
            elif calc_type in ["ssw", "SSW"]:
                atoms, energy = self.lasp_ssw_calc(atoms)
                return atoms, energy
            elif calc_type in ['MD', 'md']:
                atoms = self.lasp_md_calc(atoms)
                return atoms
            elif calc_type in ['TS', 'ts']:
                barrier = self.lasp_ts_calc(atoms)
                return barrier
            else:
                raise ValueError("No such calc type currently!!!")
            
        elif self.calculate_method in ["MACE", "Mace", "mace"]:
            if self.model_path is None:
                self.model_path = 'my_mace.model'
            if calc_type == "opt":
                atoms, energy, force = self.mace_calc(atoms)
                return atoms, energy, force
            elif calc_type in ["single-point", "single"]:
                energy = self.mace_single_calc(atoms)
                return energy
            elif calc_type in ['MD', 'md']:
                atoms = self.mace_md_calc(atoms)
                return atoms
            elif calc_type in ['TS', 'ts']:
                atoms, convereged = self.mace_ts_calc(atoms)
                return atoms,convereged
            else:
                raise ValueError("No such calc type currently!!!")
            
        elif self.calculate_method in ["NequIP", "Nequip", "nequip"]:
            if calc_type == "opt":
                atoms, energy, force = self.nequip_calc(atoms)
                return atoms, energy, force
            elif calc_type in ["single-point", "single"]:
                energy = self.nequip_single_calc(atoms)
                return energy
            else:
                raise ValueError("No such calc type currently!!!")
        
        else:
            raise ValueError("No such calculator currently!!!")

    def to_constraint(self, atoms:ase.Atoms) -> FixAtoms:
        constraint_list = atoms.constraints[0].get_indices()[0]
        print(f"The constraint list is {constraint_list}")
        if constraint_list:
            constraint = FixAtoms(mask=constraint_list)
            atoms.set_constraint(constraint)

    '''-------------MACE_calc--------------------'''
    def mace_calc(self, atoms:ase.Atoms) -> Tuple:
        from mace.calculators import MACECalculator

        calculator = MACECalculator(model_paths=self.model_path, device='cuda')
        atoms.set_calculator(calculator)

        dyn = LBFGS(atoms, trajectory='lbfgs.traj')
        dyn.run(steps = 200, fmax = 0.1)

        return atoms, atoms.get_potential_energy(), atoms.get_forces()

    def mace_single_calc(self, atoms:ase.Atoms) -> int:
        from mace.calculators import MACECalculator

        calculator = MACECalculator(model_paths=self.model_path, device='cuda')
        atoms.set_calculator(calculator)

        return atoms.get_potential_energy()
    
    def mace_md_calc(self, atoms:ase.Atoms) -> ase.Atoms:
        from mace.calculators import MACECalculator

        steps = 100
        calculator = MACECalculator(model_paths=self.model_path, device='cuda')
        atoms.set_calculator(calculator)
        dyn = Langevin(atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory='md.traj',
                           logfile='MD.log')
        dyn.run(steps)
        return atoms
    
    def mace_ts_calc(self, atoms:ase.Atoms) -> ase.Atoms:
        from mace.calculators import MACECalculator
        from ase.neb import NEB

        steps = 5000
        IS = atoms[0].copy()
        FS = atoms[1].copy()

        images = [IS]
        for _ in range(5):
            image = IS.copy()
            image.calc = MACECalculator(model_paths = self.model_path, device = "cuda")
            images.append(image)
        images.append(FS)

        for insert_image in images:
            self.to_constraint(insert_image)
        
        neb = NEB(images, allow_shared_calculator=True)
        neb.interpolate(apply_constraint = True)
        dyn = LBFGS(neb, trajectory='A2B.traj')
        dyn.run(fmax = 0.05, steps = steps)

        ts_energy = neb.get_potential_energy()
        return ts_energy, dyn.converged()
    
    '''-------------------LASP_calc---------------------------'''    
    def lasp_calc(self, atoms):
        write_arc([atoms])
        atoms.calc = LASP(task='opt', pot=self.model_path, potential='NN D3', command = LASP_COMMAND)
        energy = atoms.get_potential_energy()
        force = atoms.get_forces()
        atoms = read_arc('allstr.arc', index = -1)
        return atoms, energy, force
    
    def lasp_single_calc(self, atoms):
        write_arc([atoms])
        atoms.calc = LASP(task='single-energy', pot=self.model_path, potential='NN D3', command = LASP_COMMAND)
        energy = atoms.get_potential_energy()
        return energy
    
    def lasp_ssw_calc(self, atoms):
        write_arc([atoms])
        atoms.calc = LASP(task='long-ssw', pot=self.model_path, potential='NN D3', command = LASP_COMMAND)
        energy = atoms.get_potential_energy()
        atoms = read_arc('best.arc', index = -1)
        return atoms, energy
    
    def lasp_md_calc(self, atoms):
        steps = 100
        atoms.calc = EMT()
        dyn = Langevin(atoms, 5 * units.fs, self.temperature_K * units.kB, 0.002, trajectory='md.traj',
                           logfile='MD.log')
        dyn.run(steps)
        return atoms
    
    def lasp_ts_calc(self, atoms:List[ase.Atoms]):
        write_arc(atoms[0])
        write_arc(atoms)
        atoms[0].calc = LASP(task='TS', pot=self.model_path, potential='NN D3', command = LASP_COMMAND)
        if atoms[0].get_potential_energy() == 0:  #没有搜索到过渡态
            barrier = 0
        else:
            barrier, _ = atoms[0].get_potential_energy()

        return barrier

    '''----------------------Nequip_calc--------------------------'''
    def nequip_calc(self, atoms):
        from nequip.ase import NequIPCalculator

        calc = NequIPCalculator.from_deployed_model(model_path=self.model_path,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            species_to_type_name = {'O': 'O', self.metal_ele: self.metal_ele},
                )
        
        atoms.calc = calc
        dyn = LBFGS(atoms, trajectory='lbfgs.traj')
        dyn.run(steps = 200, fmax = 0.05)

        return atoms, atoms.get_potential_energy(), atoms.get_forces()
    
    def nequip_single_calc(self, atoms):
        from nequip.ase import NequIPCalculator

        calc = NequIPCalculator.from_deployed_model(model_path=self.model_path,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            species_to_type_name = {'O': 'O', self.metal_ele: self.metal_ele},
                )
        
        atoms.calc = calc

        return atoms.get_potential_energy()
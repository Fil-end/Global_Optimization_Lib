import os
import re

from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from ase.io.dmol import write_dmol_arc
import numpy as np

class LASP(FileIOCalculator):
    implemented_properties = ['energy', 'forces']

    # command = 'source /opt/intelstart.sh;mpirun -np 48 /data/apps/lasp/lasp 1>&2 > print-out'
    command = 'unset $(compgen -v | grep SLURM); unset SLURM_PROCID;mpirun -np 24 lasp' # on genzi

    default_parameters = dict(task='single-energy', pot=["PdO.pot"], potential='NN D3', explore='SSW', POT=None)

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='lasp', atoms=None, command=None,  **kwargs):
        """ ASE interface to LASP
        by Jianrui Geng, Based on ORCA and VASP interfaces but simplified.
        Only supports energies and gradients (no dipole moments,
        orbital energies etc.) for now.

        Key words:
		task: str, -> 'single energy', 'local-opt', 'TS', 'global-opt'(Now we can only use the first one to obtain energy and force)
       
		pot: list, -> the name of your ML potential function file, such as ['NiCHO.pot', 'B-solid.pot']
        WARNING:
        But we have no great idea to tackle the problem that some elements exist in more than one potential function files, 
        such as we provide the NiCHO.pot and NiCHN.pot .
        In this case, there is no way to  automatically point out the map of element and pot.
        
		potential: str, -> 'NN', 'D3'
		explore: str, -> 'SSW'
		multi-pot lasp.in: pot = ["NiCHO.pot", "NFOC.pot"] and POT = ' Ni NiCHO.pot\n F CFON.pot\n'

       
        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        
        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.label.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)
        
        self.atoms = atoms

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def split_pot_elements(self, pot_name):
        elements = re.findall(r'[A-Z]', pot_name)
        merged_elements = re.findall(r'[A-Z][a-z]', pot_name) 
        for i in range(len(pot_name)):
            if i + 1 < len(pot_name) and pot_name[i] in elements and pot_name[i + 1] in elements:  
                merged_elements.append(pot_name[i]) 
            elif i + 1 == len(pot_name) and pot_name[i] in elements:
                merged_elements.append(pot_name[i]) 
        unique_elements = set(merged_elements)  
        
        return unique_elements

    def write_lasp(self, atoms, **params):
        """Function to write LASP input file
    """
        with open('lasp.in', 'w') as f:
            f.write(f"potential {params['potential']}\nexplore type {params['explore']}\n")
            s = self.split_pot_elements(pot_name=params['pot'].split('.')[0])
            cmd = f"cp /dssg/home/acct-xmcao/xmcao/sychen/work/LASP/pot/{params['pot']} ./"     # on genzi
            os.system(cmd)
            f.write('%block netinfo\n')
            for i in s:
                f.write(f' {i} {params["pot"]}\n')

            f.write(f'%endblock netinfo\n')

            if params['task'] == 'single-energy':
                f.write('Run_type 5\nSSW.quick_setting 1\nSSW.Temp 200\nSSW.printevery True\nSSW.output True\nSSW.SSWsteps 0\n')
            if params['task'] == 'ssw':
                f.write('Run_type 5\nSSW.ftol 0.05\nSSW.strtol 0.05\nSSW.output T\nSSW.printevery T\nSSW.Temp 200\nSSW.SSWsteps 1\n')
            if params['task'] == 'long-ssw':
                f.write('Run_type 5\nSSW.ftol 0.05\nSSW.strtol 0.05\nSSW.output T\nSSW.printevery T\nSSW.Temp 200\nSSW.SSWsteps 300\n')
            if params['task'] == 'opt':
                f.write('Run_type 0\nSSW.ftol 0.05\nSSW.strtol 0.05\nSSW.output T\nSSW.printevery T\nSSW.Temp 200\nSSW.MaxOptstep           200\n')
            if params['task'] == 'TS':
                f.write('Run_type 2\nDESW.quick_setting 1\nDESW.task optpath\nSSW.output T\nDESW.optpath_cycle 4\nDESW.optpath_countP 8\nSSW.Rotftol 0.01\nDESW.ds 0.1\nCBD.maxstep 25\nCBD.maxcycle 40\nCBD.maxdist 0.05\nCBD.fact 0.05\nCBD.cellfact 0.5\n')
            
            if atoms.constraints:
                f.write('%block fixatom\n')
                for i in atoms.constraints[0].get_indices()+1:
                    f.write(f'{i} {i}\n')
                f.write('%endblock fixatom\n')

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters
        p.write(self.label + '.ase')
        p['label'] = self.label
        atoms.pbc = True
        write_dmol_arc('input.arc', [atoms])
        self.write_lasp(atoms, **p)

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open('input.arc') as fd:
            positions = []
            symbols = []
            lines = fd.readlines()
        for line in lines:
            Line = line.split()
            if len(Line) == 7:
                cell_params = Line[1:]
            if len(Line) == 10:
                positions.append(Line[1:4])
                symbols.append(Line[0])
       # positions = np.array(positions ,dtype=float)
        #cell = [float(param) for param in cell_params]
        #atoms = Atoms(symbols=symbols, positions=positions, pbc=True, cell=cell)
        
        self.parameters = Parameters.read(self.label + '.ase')
        self.read_results()

    def read_results(self):
        self.read_forces_energy()

    def _split_allfor(self):

        order = []
        contents = []
        space_number = 1
        stress_number = 1
        title_number =1

        with open('allfor.arc', 'r') as fd:
            lines = fd.readlines()

            for i, line in enumerate(lines):
                if line.find('For') >= 0:
                    order.append((i))
            if len(order) > 1:
                line_numbers = order[-1] - order[-2]
                size = line_numbers
                nrow = len(lines)
                start = 0
                end = size
                for i in range(nrow//size):

                    batch_name = 'force.tmp'
                    with open(batch_name, 'w') as o:
                        o.write(''.join(lines[start:end]))
                    start = start + size
                    end = end + size
                return 1
            else:
                return 0
               

    def read_forces_energy(self):
        """Read Forces and Energy from LASP output file."""  
        p = self.parameters
        if p['task'] in ['single-energy','opt', 'ssw', 'long-ssw']:
            gradients = []
            flag = self._split_allfor()
            if flag == 1:
                filename = "force.tmp"
            else:
                filename = "allfor.arc"

            with open(filename, 'r') as fd:
                lines = fd.readlines()[:]
                for i, line in enumerate(lines):
                    Line = line.split()

                    if len(Line) == 4:
                        self.results['energy'] = float(Line[-1]) #* Hartree

                    if len(Line) == 3:
                        gradients.append(Line)
                        self.results['forces'] = np.array(gradients, dtype=float) #* Hartree / Bohr
                    
        elif p['task'] == 'TS':
           with open('lasp.out','r') as fd:
                line = fd.readline()
                pattern1 = "BarrierandReactionenergy"
                pattern2 = "TSsearchfails"
                pattern3 = "Fail"
                pattern4 = "SameStr-pairfound"
                pattern5 = "DESW::Failtolinkpathway,earlybreak"
                while line:
                    newline = line.replace(" ","")
                    if (newline[0:len(pattern1)] == pattern1):
                        line = fd.readline()
                        self.results['energy'] = [float(line.split()[3]) - float(line.split()[2]), float(line.split()[4]) - float(line.split()[2])]
                        newline = line.replace(" ", "")
                    elif (newline[0:len(pattern2)] == pattern2):
                        line = fd.readline()
                        self.results['energy'] = 0
                        newline = line.replace(" ", "")
                    elif (newline[0:len(pattern3)] == pattern3):
                        line = fd.readline()
                        self.results['energy'] = 0
                        newline = line.replace(" ", "")
                    elif (newline[0:len(pattern4)] == pattern4):
                        line = fd.readline()
                        self.results['energy'] = 0
                        newline = line.replace(" ", "")
                    elif (newline[0:len(pattern5)] == pattern5):
                        line = fd.readline()
                        self.results['energy'] = 0
                        newline = line.replace(" ", "")
                    line = fd.readline()












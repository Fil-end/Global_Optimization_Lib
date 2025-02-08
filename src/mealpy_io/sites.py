from dataclasses import dataclass
from typing import List

from ase import Atoms, Atom
from ase.data import chemical_symbols
from ase.geometry import wrap_positions
import numpy as np
from numpy.linalg import norm
from scipy.spatial import Delaunay, distance_matrix

from periodic_table import METALLIST

D_LIST = [1.5, 1.3, 0.6]

@dataclass
class Sites:
    '''The Sites class is current used for the free atoms sites selection
     for global optimization
     
     Args:
        atoms(Atoms): fixed surface atoms
        ele_list(List): atoms for global optimization
    '''

    atoms: Atoms
    ele_list: List

    def __post_init__(self):
        self.sub_z = max(self.atoms.positions[:, 2])
        self.metal_ele = self._metal_ele()
        print(self.metal_ele)

    def __call__(self) -> Atoms:
        return self.choose_ads_site()
        
    def choose_ads_site(self) -> Atoms:
        new_state = self.atoms.copy()
        self.forbidden_site = []
        for ele_idx in self.ele_list:
            ele = chemical_symbols[ele_idx]
            empty_sites = self.empty_sites(new_state)
            atom_site = empty_sites[np.random.randint(len(empty_sites))]

            atom = Atom(ele, (atom_site[0],
                              atom_site[1], 
                              atom_site[2] + D_LIST[int(atom_site[3]) - 1 ] * 1.5))
            new_state += atom
            self.forbidden_site.append(atom_site.tolist())

        self.forbidden_site = []
        return new_state
    
    def _metal_ele(self) -> None:
        '''In self.ele_list, metal element should be former than nonmetal element'''
        metal_list = []

        try:
            for ele_idx in self.ele_list:
                if chemical_symbols[ele_idx] in METALLIST:
                    metal_list.append(chemical_symbols[ele_idx])
        except:
            raise ValueError('Your input element index should in (1-103)')
        
        return list(set(metal_list))

    def empty_sites(self, atoms:Atoms) -> np.asarray:
        surf_atoms = self.get_surf_list(atoms)
        sub_sites = self.in_cell_sites(atoms, site_type='sub')

        empty_sites = []

        if not surf_atoms:
            # Sub layer is empty
            empty_sites = sub_sites
        else:
            for sub_site in sub_sites:
                to_other_ele_dis_1 = []
                for atom_idx in surf_atoms:
                    distance = self.distance(atoms.get_positions()[atom_idx], 
                                             sub_site[0:3] + np.array([0, 0, D_LIST[int(sub_site[3]) - 1 ]]))
                    to_other_ele_dis_1.append(distance)

                if min(to_other_ele_dis_1) >= 1.5:
                    empty_sites.append(sub_site)
            # All sub layer sites has been occupied
            if not empty_sites:
                surf_sites = self.in_cell_sites(atoms, site_type='surf')
                layer_atoms = self.get_layer_list(atoms)
                if layer_atoms:
                    for surf_site in surf_sites:
                        to_other_ele_dis_2 = []
                        for atom_idx in layer_atoms:
                            distance = self.distance(atoms.get_positions()[atom_idx], 
                                                     surf_site[0:3]+ np.array([0, 0, self.D_LIST[int(surf_site[3]) - 1 ]]))
                            to_other_ele_dis_2.append(distance)
                        if min(to_other_ele_dis_2) >= 1.5:
                            empty_sites.append(surf_site)
                else:
                    empty_sites = surf_sites
                    
        return empty_sites
    
    def in_cell_sites(self, atoms:Atoms, site_type = 'surf') -> np.asarray:
        primitive_cell = atoms.cell
        # atoms = atoms * (3,3,1)
        # self.rectify_position(atoms)

        total_sites = []
        if site_type == 'surf':
            total_surf_sites = self.get_surf_sites(atoms)
            for surf_site in total_surf_sites:
                w_p = wrap_positions([surf_site[0:3].tolist()], primitive_cell.tolist(), pbc=[1,1,0])
                if '%.2f' % surf_site[0] == '%.2f' % w_p.tolist()[0][0] and \
                    '%.2f' % surf_site[1] == '%.2f' % w_p.tolist()[0][1] and \
                    '%.2f' % surf_site[2] == '%.2f' % w_p.tolist()[0][2]:
                    total_sites.append(surf_site.tolist())

        elif site_type == 'sub':
            total_sub_sites = self.get_sub_sites(atoms)
            for sub_site in total_sub_sites:
                w_p = wrap_positions([sub_site[0:3].tolist()], primitive_cell.tolist(), pbc=[1,1,0])
                if '%.2f' % sub_site[0] == '%.2f' % w_p.tolist()[0][0] and \
                    '%.2f' % sub_site[1] == '%.2f' % w_p.tolist()[0][1] and \
                    '%.2f' % sub_site[2] == '%.2f' % w_p.tolist()[0][2]:
                    total_sites.append(sub_site.tolist())

        total_sites = np.array([site for n,site in enumerate(total_sites) \
                                if site not in total_sites[:n] and site not in self.forbidden_site])
        return np.array(total_sites)
    
    '''----------------------- get atom index list ----------------------------------'''
    def get_layer_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [self.sub_z + 2.0, self.sub_z + 10.0])
    
    def get_surf_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [self.sub_z + 1.0, self.sub_z + 3.0])
    
    def get_sub_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [self.sub_z - 1.0, self.sub_z + 1.0])
    
    def label_atoms(self, atoms:Atoms, zRange:List) -> List[int]:
        myPos = atoms.get_positions()
        return [
            i for i in range(len(atoms)) \
            if min(zRange) < myPos[i][2] < max(zRange)
        ]
    
    '''----------------------- get surface sites ----------------------------------'''
    def get_surf_sites(self, atoms:Atoms) -> np.asarray:
        surfList = self.get_surf_list(atoms)

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if (i not in surfList) or \
                  surf[i].symbol not in self.metal_ele]]
        
        surf_sites = self.get_sites(surf)

        return surf_sites
    
    def get_sub_sites(self, atoms:Atoms) -> np.asarray:
        subList = self.get_sub_list(atoms)

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if (i not in subList) or \
                 sub[i].symbol not in self.metal_ele]]

        sub_sites= self.get_sites(sub)
        return sub_sites

    def get_sites(self, atoms:Atoms) -> np.asarray:
        if len(atoms) == 0:
            return None
        elif len(atoms) == 1:
            sites = []
            for _ in range(2):
                sites.append(np.array([atoms.get_positions()[0][0],atoms.get_positions()[0][1],atoms.get_positions()[0][2], 1, 0]))
            return np.array(sites)
        elif len(atoms) == 2:
            sites = []
            for atom in atoms:
                sites.append(np.append(atom.position, [1, 0]))
            sites.append(np.array([(atoms.get_positions()[0][0] + atoms.get_positions()[1][0]) / 2,
                                   (atoms.get_positions()[0][1] + atoms.get_positions()[1][1]) / 2,
                                   (atoms.get_positions()[0][2] + atoms.get_positions()[1][2]) / 2,
                                   2, 0]))
            return np.array(sites)
        elif len(atoms) >= 3:
            atop = atoms.get_positions()
            pos_ext = atoms.get_positions()
            tri = Delaunay(pos_ext[:, :2])
            pos_nodes = pos_ext[tri.simplices]

            bridge_sites = []
            hollow_sites = []

            for i in pos_nodes:
                if (self.distance(i[0], i[1])) < 3.0:
                    bridge_sites.append((i[0] + i[1]) / 2)
                else:
                    hollow_sites.append((i[0] + i[1]) / 2)

                if (self.distance(i[2], i[1])) < 3.0:
                    bridge_sites.append((i[2] + i[1]) / 2)
                else:
                    hollow_sites.append((i[2] + i[1]) / 2)

                if (self.distance(i[0], i[2])) < 3.0:
                    bridge_sites.append((i[0] + i[2]) / 2)
                else:
                    hollow_sites.append((i[0] + i[2]) / 2)

            top_sites = np.array(atop)
            hollow_sites = np.array(hollow_sites)
            bridge_sites = np.array(bridge_sites)

            sites_1 = []
            total_sites = []

            for i in top_sites:
                sites_1.append(np.transpose(np.append(i, 1)))
            for i in bridge_sites:
                sites_1.append(np.transpose(np.append(i, 2)))
            for i in hollow_sites:
                sites_1.append(np.transpose(np.append(i, 3)))
            for i in sites_1:
                total_sites.append(np.append(i, 0))

            total_sites = np.array(total_sites)

        return total_sites
    
    def distance(self, x1: np.asarray, x2:np.asarray) -> float:
        return norm(x1 - x2)

from dataclasses import dataclass
import math
from math import cos, sin
import os
import timeout_decorator
from typing import List, Tuple, Optional

# Chemical Structure operation platform
from ase import Atom, Atoms
from ase.build import fcc100, molecule
from ase.constraints import FixAtoms
from ase.data import covalent_radii
from ase.geometry.analysis import Analysis
from ase.io import Trajectory
from ase.visualize.plot import plot_atoms
# Tensor operation package
from einops import rearrange
# Reinforcement learning packages
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
# Deep learning platform
import torch
import torch.nn as nn

# from EDRL.io.lasp import write_arc
# from EDRL.tools.utils import to_pad_the_array
from calc import Calculator
# from EDRL.ts.gpr import GPR
# import EDRL.utils.Painn_utils as Painn

bottom_z = 10.7
deep_z = 12.1
sub_z = 16.1
surf_z = 18.1
layer_z = 21.1

fluct_d_metal = 1.0
fluct_d_layer = 3.0

r_O = covalent_radii[8]
r_Pd = covalent_radii[46]

d_O_Pd = r_O + r_Pd
d_O_O = 2 * r_O
d_Pd_Pd = 2 * r_Pd

'''
    Actions:
        Type: Discrete(8)
        Num   Action
        0     ADS
        1     Translation
        2     R_Rotation
        3     L_Rotation
        4     Min
        5     Diffusion
        6     Drill
        7     Dissociation

        '''
# 设定动作空间
ACTION_SPACES = ['ADS', 'Translation', 'R_Rotation', 'L_Rotation', 'MD', 'Diffusion', 'Drill', 'Dissociation', 'Desportion']

# 创建MCT环境
@dataclass
class MCTEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    model_path:str                      # loading the model
    initial_slab:Optional[Atoms] = None
    reaction_H:float = 0.617            #/eV
    reaction_n:float = 32
    delta_s:float  = -0.371             #/eV
    save_dir:str = 'save_dir'
    timesteps:int = 200                 # 定义episode的时间步上限
    temperature:float = 473.15          # 设定环境温度为473.15 K，定义热力学能
    k:float = 8.6173324e-05             # eV/K
    max_energy_profile:float = 0.5
    max_energy_span:float = 10.0
    convergence:float = 0.005
    save_every:int = 1
    ts_method:str = "BEP"
    use_GNN_description:bool = True    # use Painn description
    use_kinetic_penalty:bool = False 
    cutoff:int = 4.0                    # Painn paras
    hidden_state_size:int = 50          # embedding_output_dim and the hidden_dim overall the Painn
    embedding_size:int = 50
    num_interactions:int = 3
    calculator_method:str = 'MACE'
    metal_ele:str = 'Ag'
    max_observation_atoms:Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.initial_slab, Atoms):
            self.initial_slab = self._generate_initial_slab()  # 设定初始结构
        self.calculator = Calculator(calculate_method = self.calculator_method,
                                     model_path = self.model_path)

        # self.E_O2 = self.add_mole("O2")
        # self.E_O3 = self.add_mole("O3")

        self.episode = 0  # 初始化episode为0
        self.timestep = 0  # 初始化时间步数为0

        self.H = self.reaction_H * self.reaction_n

        self.range = [0.9, 1.1]
        self.d_list = [1.5, 1.3, 0.6]
        self.reward_threshold = 0
        # 保存history,plots,trajs,opt
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.history_dir = os.path.join(self.save_dir, 'history')
        self.plot_dir = os.path.join(self.save_dir, 'plots')

        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # 初始化history字典
        self.history = {}
        # 记录不同吸附态（Pd64O,Pd64O2......Pd64O16）时的能量以及结构
        # self.adsorb_history = {}

        if self.max_observation_atoms is None:
            self.max_observation_atoms = 2 * len(self.initial_slab)

        # 定义动作空间
        # self.action_space = spaces.Discrete(len(ACTION_SPACES))
        # 定义状态空间
        # self.observation_space = self.get_observation_space()

        # 一轮过后重新回到初始状态
        # self.reset()

        return

    def step(self, action):
        barrier = 0
        self.steps = 100  # 定义优化的最大步长
        reward = 0  # 定义初始奖励为0

        self.action_idx = action
        RMSD_similar = False
        kickout = False
        action_done = True

        episode_over = False  # 与done作用类似
        target_get = False

        self.atoms, _, previous_energy = self.state

        assert self.action_space.contains(self.action_idx), "%r (%s) invalid" % (
            self.action_idx,
            type(self.action_idx),
        )
        self.layer_atom, self.surf_atom, self.sub_atom,self.deep_atom = self.get_atom_info(self.atoms)
        
        # 定义层之间的平动弛豫

        '''——————————————————————————————————————————以下是动作选择————————————————————————————————————————————————————————'''
        if self.action_idx == 0:
            self.atoms = self.choose_ads_site(self.atoms)

        elif self.action_idx == 1:
            self._translate()

        elif self.action_idx == 2:
            self._rotate(self.atoms, 9)

        elif self.action_idx == 3:
            self._rotate(self.atoms, -9)

        elif self.action_idx == 4:
            self._md()

        #------------The above actions are muti-actions and the following actions contain single-atom actions--------------------------------
        elif self.action_idx == 5:  # 表面上氧原子的扩散，单原子行为
            self.atoms, action_done = self._diffuse(self.atoms)

        elif self.action_idx == 6:  # 表面晶胞的扩大以及氧原子的钻洞，多原子行为+单原子行为
            self.atoms, action_done = self._drill(self.atoms)

        elif self.action_idx == 7:  # 氧气解离
            self.atoms, action_done = self._dissociate(self.atoms)

        elif self.action_idx == 8:
            self.atoms, action_done = self._desorb(self.atoms)
            
        else:
            print('No such action')

        self.timestep += 1
        if action_done:
            reward -= 1

        previous_atom = self.trajectories[-1]
        
                # 优化该state的末态结构以及next_state的初态结构
        self.to_constraint(self.atoms)
        write_arc([self.atoms], name = "chk_pt.arc")
        self.atoms, current_energy, current_force = self.calculator(self.atoms)
        self.atoms, del_list = self.del_env_adsorbate(self.atoms)

        if del_list:
            self.atoms, current_energy, _ = self.calculator(self.atoms)

        print(f"The action is {action}, num O2 is {self.n_O2}, num O3 is {self.n_O3}, \
                  PdOx x= {len(self.atoms) - len(self.initial_slab)}")

        current_energy = current_energy + self.n_O2 * self.E_O2 + self.n_O3 * self.E_O3
        print(f"The current_energy is {current_energy}")

        # Here is the kickout part of the project
        # kickout the structure if too similar
        if self.timestep > 11:
            if self.RMSD(self.atoms, self.trajectories[-10])[0] and (current_energy - self.history['energies'][-10]) > 0: 
                RMSD_similar = True
                kickout = True

        # kickout the structure if too bad or broken
        elif self.get_bond_info(self.atoms) or self._env_O(self.atoms) or \
            current_energy - previous_energy > 30:   # 如果结构过差，将结构kickout
            kickout = True

        # (1) action not done.
        # (2) the structure is too bad to be kicked out.
        # (3) one specific action has repeated too many times and is none
        #     sense or side effects.
        # The agent will get penalty
        if not action_done:
            reward += -2

        if kickout:
            self.atoms = previous_atom
            current_energy = previous_energy
            self.n_O2, self.n_O3 = self.history['adsorbates'][-1]
            reward += -5

        if action == 0 and not self.atoms == previous_atom:
            current_energy = current_energy - self.delta_s

        if action == 8 and not self.atoms == previous_atom:
            current_energy = current_energy + self.delta_s

        relative_energy = current_energy - previous_energy
        if relative_energy > 5:
            reward += -1
        else:
            reward += self.get_reward_sigmoid(relative_energy)

        if self.timestep > 6:
            current_action_list = self.history['actions'][-5:]
            result = all(x == current_action_list[0] for x in current_action_list)
            if result and action == current_action_list[0] and (RMSD_similar and relative_energy >= 0):
                self.repeat_action += 1
                reward -= self.repeat_action * 1
            elif result and action != current_action_list[0]:
                self.repeat_action = 0
            
        self.RMSD_list.append(self.RMSD(self.atoms, previous_atom)[1])

        if self.timestep > 6:
            current_action_list = self.history['actions'][-5:]
            result = all(x == current_action_list[0] for x in current_action_list)
            if result and action == current_action_list[0] and (RMSD_similar and relative_energy >= 0):
                self.repeat_action += 1
                reward -= self.repeat_action * 1
            elif result and action != current_action_list[0]:
                self.repeat_action = 0
        
        if self.atoms == previous_atom:
            self.record_TS(previous_energy, barrier)
        else:
            barrier = self.check_TS(previous_atom, self.atoms, previous_energy, current_energy, action) 
            if not self.use_kinetic_penalty:
                reward -= barrier / (self.H * self.k * self.temperature)
            else:
                reward -= self._kinetic_penalty(barrier)
      
        current_structure = self.atoms.get_positions()

        self.energy = current_energy
        self.force = current_force

        self.pd = nn.ZeroPad2d(padding = (0,0,0,self.max_observation_atoms-len(self.atoms.get_positions())))

        observation = self.get_obs()  # 能观察到该state的结构与能量信息

        self.state = self.atoms, current_structure, current_energy

        # Update the history for the rendering
        self.history, self.trajectories = self.update_history(action, kickout)
        
        # Here is the bad termination cases
        if self.exist_too_short_bonds(self.atoms) or self.energy - self.initial_energy > \
            len(self.atoms) * self.max_energy_profile or relative_energy > self.max_RE:
            reward -= 0.5 * self.timesteps
            episode_over = True
        
        elif self.timestep > 11:
            if self.atoms == self.trajectories[-10]:
                episode_over = True
                reward -= 0.5 * self.timesteps
                
        if -1.5 * relative_energy > self.max_RE:
            self.max_RE = -1.5 * relative_energy

        # 当步数大于时间步，停止，且防止agent一直选取扩散或者平动动作
        # 设置惩罚下限 
        if self.timestep >= self.timesteps or self.episode_reward <= self.reward_threshold or \
            len(self.history['actions']) - 1 >= self.total_steps:   
            episode_over = True

        if relative_energy >= 0:
            reward -= 1.0       # 每经历一步timesteps, -1.0

        # Here is the good termination cases
        if self.use_kinetic_penalty:
            termination = (current_energy - self.initial_energy) <= -self.H
        else:
            termination = ((current_energy - self.initial_energy) <= -self.H and (abs(current_energy - previous_energy) < self.min_RE_d \
                and abs(current_energy - previous_energy) > 0.0001))  and self.RMSD_list[-1] < 0.5
            
        if termination:     
            episode_over = True
            target_get = True
            reward += self.H -(self.energy - self.initial_energy + self.H) /(self.k * self.temperature)
        
        self.history['reward'] = self.history['reward'] + [reward]
        self.episode_reward += reward

        if episode_over:
            self.episode += 1
            if self.episode % self.save_every == 0 or target_get:
                self.save_episode()
                self.plot_episode()

        return observation, reward, episode_over, [target_get, action_done]


    def save_episode(self):
        save_path = os.path.join(self.history_dir, '%d.npz' % self.episode)
        np.savez_compressed(
            save_path,
            
            initial_energy=self.initial_energy,
            energies=self.history['energies'],
            actions=self.history['actions'],
            structures=self.history['structures'],
            timesteps=self.history['timesteps'],
            forces = self.history['forces'],
            reward = self.history['reward'],

            ts_energy = self.TS['energies'],
            ts_timesteps = self.TS['timesteps'],
            barriers = self.TS["barriers"],

            episode_reward = self.episode_reward,

        )
        return

    def plot_episode(self):
        save_path = os.path.join(self.plot_dir, '%d.png' % self.episode)

        energies = np.array(self.history['energies'])
        actions = np.array(self.history['actions'])

        plt.figure(figsize=(30, 30))
        plt.xlabel('steps')
        plt.ylabel('Energies')
        plt.plot(energies, color='blue')

        for action_index in range(len(ACTION_SPACES)):
            action_time = np.where(actions == action_index)[0]
            plt.plot(action_time, energies[action_time], 'o',
                     label=ACTION_SPACES[action_index])

        # plt.scatter(self.TS['timesteps'], self.TS['energies'], label='TS', marker='x', color='g', s=180)
        # plt.scatter(self.adsorb_history['timesteps'], self.adsorb_history['energy'], label='ADS', marker='p', color='black', s=180)
        plt.legend(loc='upper left')
        plt.savefig(save_path, bbox_inches='tight')
        return plt.close('all')

    def reset(self):
        self.H = self.reaction_H * self.reaction_n

        self.n_O2 = 2000
        self.n_O3 = 0

        self.atoms = self._generate_initial_slab()

        self.to_constraint(self.atoms)
        self.atoms, self.initial_energy, self.initial_force= self.calculator(self.atoms)
        self.initial_energy = self.initial_energy + self.n_O2 * self.E_O2 + self.n_O3 * self.E_O3

        self.action_idx = 0
        self.episode_reward = 0.5 * self.timesteps
        if self.use_kinetic_penalty:
            self.episode_reward += self.H
        self.timestep = 0
        
        # 标记可以自由移动的原子
        self.fix_atoms = self.get_constraint(self.atoms).get_indices()
        self.free_atoms = list(set(range(len(self.atoms))) - set(self.fix_atoms))
        self.len_atom = len(self.free_atoms)

        self.total_steps = self.timesteps
        self.max_RE = 3
        self.min_RE_d = self.convergence * self.len_atom
        self.repeat_action = 0

        self.ads_list = []
        for _ in range(self.n_O2):
            self.ads_list.append(2)

        self.pd = nn.ZeroPad2d(padding = (0,0,0, self.max_observation_atoms - len(self.atoms.get_positions())))

        self.trajectories = []
        self.RMSD_list = []
        self.RMSD_threshold = 0.5
        self.trajectories.append(self.atoms.copy())

        self.TS = {}
        self.TS['energies'] = [0.0]
        self.TS['barriers'] = [0.0]
        self.TS['timesteps'] = [0]

        results = ['energies', 'actions', 'structures', 'timesteps', 'forces', 'scaled_structures', 'real_energies', 'reward']
        for item in results:
            self.history[item] = []
        self.history['energies'] = [0.0]
        self.history['real_energies'] = [0.0]
        self.history['actions'] = [0]
        self.history['forces'] = [to_pad_the_array(self.initial_force, max_len = self.max_observation_atoms, position = True)]
        self.history['structures'] = [to_pad_the_array(self.atoms.get_positions(),
                                                       max_len = self.max_observation_atoms,).flatten()]
        self.history['scaled_structures'] = [to_pad_the_array(self.atoms.get_scaled_positions()[self.free_atoms, :],
                                                              max_len = self.max_observation_atoms,).flatten()]
        self.history['timesteps'] = [0]
        self.history['adsorbates'] = [(self.n_O2, self.n_O3)]
        self.history['reward'] = []

        self.state = self.atoms, self.atoms.positions, self.initial_energy

        observation = self.get_obs()

        return observation

    def render(self, mode='rgb_array'):

        if mode == 'rgb_array':
            # return an rgb array representing the picture of the atoms

            # Plot the atoms
            fig, ax1 = plt.subplots()
            plot_atoms(self.atoms.get_scaled_positions(),
                       ax1,
                       rotation='48x,-51y,-144z',
                       show_unit_cell=0)

            ax1.set_ylim([-1, 2])
            ax1.set_xlim([-1, 2])
            ax1.axis('off')
            ax2 = fig.add_axes([0.35, 0.85, 0.3, 0.1])

            # Add a subplot for the energy history overlay
            ax2.plot(self.history['timesteps'],
                     self.history['energies'])

            if len(self.TS['timesteps']) > 0:
                ax2.plot(self.TS['timesteps'],
                         self.TS['energies'], 'o', color='g')

            ax2.set_ylabel('Energy [eV]')

            # Render the canvas to rgb values for the gym render
            plt.draw()
            renderer = fig.canvas.get_renderer()
            x = renderer.buffer_rgba()
            img_array = np.frombuffer(x, np.uint8).reshape(x.shape)
            plt.close()

            # return the rendered array (but not the alpha channel)
            return img_array[:, :, :3]

        else:
            return

    def close(self):
        return

    def get_observation_space(self):
        if self.use_GNN_description:
            observation_space = spaces.Dict({'structures':
            spaces.Box(
                low=-1,
                high=2,
                shape=(self.max_observation_atoms, ),
                dtype=float
            ),
            'energy': spaces.Box(
                low=-50.0,
                high=5.0,
                shape=(1,),
                dtype=float
            ),
            'force':spaces.Box(
                low=-2,
                high=2,
                shape=(self.max_observation_atoms, ),
                dtype=float
            ),
            'TS': spaces.Box(low = -0.5,
                                    high = 1.5,
                                    shape = (1,),
                                    dtype=float),
        })
        else:
            observation_space = spaces.Dict({'structures':
                spaces.Box(
                    low=-1,
                    high=2,
                    shape=(self.max_observation_atoms * 3, ),
                    dtype=float
                ),
                'energy': spaces.Box(
                    low=-50.0,
                    high=5.0,
                    shape=(1,),
                    dtype=float
                ),
                'force':spaces.Box(
                    low=-2,
                    high=2,
                    shape=(self.max_observation_atoms * 3, ),
                    dtype=float
                ),
                'TS': spaces.Box(low = -0.5,
                                        high = 1.5,
                                        shape = (1,),
                                        dtype=float),
            })
        return observation_space

    def get_obs(self):
        observation = {}
        if self.use_GNN_description:
            observation['structure_scalar'], observation['structure_vector'] = self._use_Painn_description(self.atoms)
            return observation['structure_scalar'], observation['structure_vector']
        else:
            observation['structure'] = self._use_MLP(self.atoms)
            return observation['structure']

    def update_history(self, action_idx, kickout):
        self.trajectories.append(self.atoms.copy())
        self.history['timesteps'] = self.history['timesteps'] + [self.history['timesteps'][-1] + 1]
        self.history['energies'] = self.history['energies'] + [self.energy - self.initial_energy]
        self.history['actions'] = self.history['actions'] + [action_idx]
        self.history['structures'] = self.history['structures'] + [to_pad_the_array(self.atoms.get_positions(), 
                                                                                    max_len = self.max_observation_atoms, position = True).flatten()]
        self.history['scaled_structures'] = self.history['scaled_structures'] + [to_pad_the_array(self.atoms.get_scaled_positions()[self.free_atoms, :],
                                                                                                  max_len = self.max_observation_atoms, position = True).flatten()]
        self.history['adsorbates'] = self.history['adsorbates']+ [(self.n_O2, self.n_O3)]
        if not kickout:
            self.history['real_energies'] = self.history['real_energies'] + [self.energy - self.initial_energy]

        return self.history, self.trajectories

    def transition_state_search(self, previous_atom:Atoms, current_atom:Atoms, 
                                previous_energy:float, current_energy:float, action:int) -> float:
        relative_energy = min(current_energy - previous_energy, 5)
        if action in [0, 8]:
            barrier = current_energy - previous_energy + 0.4
        elif action == 4:
            barrier = self._md_barrier(previous_energy)

        if self.ts_method in ["DESW", "Desw", "desw"]:
            self.to_constraint(previous_atom)
            barrier = self.calculator([previous_atom, current_atom], calc_type = 'ts')

        elif self.ts_method in ["SELLA","Sella", "sella"]:
            from sella import Sella, Constraints
            from mace.calculators import MACECalculator
            cons = Constraints(previous_atom)

            cons.fix_translation(self.fix_list)
            previous_atom.calc = MACECalculator(model_paths=self.model_path, device='cuda')

            dyn = Sella(
                    previous_atom,
                    constraints=cons,
                    trajectory='test_mace.traj',
                )
            dyn.run(1e-3, 1000)

            ts_energy = previous_atom.get_potential_energy()
            barrier = ts_energy - previous_energy

        elif self.ts_method in ["BEP", "Bep", "bep"]:
            if relative_energy > 5.0:
                print(f"The current action_idx is {action}, relative_energy is \
                       {relative_energy}, and the structure may broken!!!!!")
                write_arc([self.atoms], name = "broken.arc")
                current_energy = previous_energy + 5.0
                
            if action == 1:
                barrier =  0.258 * relative_energy + 2.3616

            elif action == 2 or action == 3:
                barrier = math.log(1 + pow(math.e, relative_energy), math.e)
            elif action == 5:
                barrier = 0.4704 * relative_energy + 0.9472
            elif action == 6:
                barrier = 0.6771 * relative_energy + 0.7784
            elif action == 7:
                barrier = 0.88 * relative_energy + 0.65

        elif self.ts_method in ['GPR', 'Gaussian Process Regression']:
            barrier = GPR(relative_energy = relative_energy, action = action)

        barrier = min(max(relative_energy, barrier,0), 5)
        
        if barrier > 1.0 and self.use_kinetic_penalty == 'strict' and len(previous_atom) == len(current_atom) and self.in_zeolite == False:
            ts_energy, converged = self.calculator(atoms = [previous_atom, current_atom], calc_type ='ts')
            if converged:
                ts_energy = ts_energy + self.n_O2 * self.E_O2 + self.n_O3 * self.E_O3
                barrier = ts_energy - previous_energy
                print(f"relative energy is {relative_energy}")
                print(f"The modified barrier is {barrier}")
                if barrier > 10.0:
                    raise ValueError("The ts energy is too high !!!!!!!!!")

        ts_energy = previous_energy + barrier
        print(f"The current barrier is {barrier}")
        return barrier, ts_energy


    def check_TS(self, previous_atom: Atoms, current_atom: Atoms, 
                 previous_energy:float, current_energy:float, action: int) -> float:
        barrier, ts_energy = self.transition_state_search(previous_atom, current_atom, previous_energy, current_energy, action)
        self.record_TS(barrier, ts_energy)
        return barrier

    def record_TS(self, barrier:float, ts_energy:float) -> None:
        self.TS['barriers'].append(barrier)
        self.TS['energies'].append(ts_energy - self.initial_energy)
        self.TS['timesteps'].append(self.history['timesteps'][-1] + 1)
        return
    
    def choose_ads_site(self, state:Atoms) -> Atoms:
        add_total_sites = []
        add_total_norm_vector = []
        
        total_s, total_plane_normal_vector = self.get_surf_sites(state)

        for ads_sites_index in range(len(total_s)):
            ads_sites = total_s[ads_sites_index]
            ads_plane_norm_vector = total_plane_normal_vector[ads_sites_index]

            layer_O = self.layer_O_atom_list(state)
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    distance = self.distance(ads_sites[0] + ads_plane_norm_vector[0] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[1] + ads_plane_norm_vector[1] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[2] + ads_plane_norm_vector[2] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             state.get_positions()[i][0],
                                             state.get_positions()[i][1], 
                                             state.get_positions()[i][2])
                    
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                add_total_sites.append(ads_sites)
                add_total_norm_vector.append(ads_plane_norm_vector)
        
        if add_total_sites:
            ads_index = np.random.randint(len(add_total_sites))
            ads_site = add_total_sites[ads_index]
            target_ads_norm_vector = add_total_norm_vector[ads_index]
  
            new_state = state.copy()
            choosed_adsorbate = np.random.randint(len(self.ads_list))
            ads = self.ads_list[choosed_adsorbate]

            del self.ads_list[choosed_adsorbate]
            
            if ads:
                if ads == 2:
                    self.n_O2 -= 1
                    O1 = Atom('O', (ads_site[0] + target_ads_norm_vector[0] * self.d_list[int(ads_site[3]) - 1 ], 
                                    ads_site[1] + target_ads_norm_vector[1] * self.d_list[int(ads_site[3]) - 1 ], 
                                    ads_site[2] + target_ads_norm_vector[2] * self.d_list[int(ads_site[3]) - 1 ]))
                    
                    O2 = Atom('O', (ads_site[0] + target_ads_norm_vector[0] * self.d_list[int(ads_site[3]) - 1 ],
                                    ads_site[1] + target_ads_norm_vector[1] * self.d_list[int(ads_site[3]) - 1 ], 
                                    ads_site[2] + target_ads_norm_vector[2] * self.d_list[int(ads_site[3]) - 1 ] + 1.21))
                    new_state = new_state + O1
                    new_state = new_state + O2

                elif ads == 3:
                    self.n_O3 -= 1
                    O1 = Atom('O', (ads_site[0] + target_ads_norm_vector[0] * self.d_list[int(ads_site[3]) - 1 ],
                                    ads_site[1] + target_ads_norm_vector[1] * self.d_list[int(ads_site[3]) - 1 ],
                                    ads_site[2] + target_ads_norm_vector[2] * self.d_list[int(ads_site[3]) - 1 ]))
                    
                    O2 = Atom('O', (ads_site[0] + target_ads_norm_vector[0] * self.d_list[int(ads_site[3]) - 1 ], 
                                    ads_site[1] + target_ads_norm_vector[1] * self.d_list[int(ads_site[3]) - 1 ] + 1.09,
                                    ads_site[2] + target_ads_norm_vector[2] * self.d_list[int(ads_site[3]) - 1 ] + 0.67))
                    
                    O3 = Atom('O', (ads_site[0] + target_ads_norm_vector[0] * self.d_list[int(ads_site[3]) - 1 ],
                                    ads_site[1] + target_ads_norm_vector[1] * self.d_list[int(ads_site[3]) - 1 ] - 1.09,
                                    ads_site[2] + target_ads_norm_vector[2] * self.d_list[int(ads_site[3]) - 1 ] + 0.67))
                    
                    new_state = new_state + O1
                    new_state = new_state + O2
                    new_state = new_state + O3

            return new_state
        else:
            return state
    
    def _desorb(self, state:Atoms) -> Tuple[Atoms, bool]:
        action_done = True
        new_state = state.copy()

        ana = Analysis(new_state)
        OOBonds = ana.get_bonds('O','O',unique = True)

        desorblist = []

        if OOBonds[0]:
            desorb = self.to_desorb_adsorbate(new_state)
            if len(desorb):
                if len(desorb) == 2:
                    self.ads_list.append(2)
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                elif len(desorb) == 3:
                    self.ads_list.append(3)
                    desorblist.append(desorb[0])
                    desorblist.append(desorb[1])
                    desorblist.append(desorb[2])

                del new_state[[i for i in range(len(new_state)) if i in desorblist]]

                if len(desorb) == 2:
                    self.n_O2 += 1

                elif len(desorb) == 3:
                    self.n_O3 += 1
            else:
                action_done = False
        action_done = False

        return new_state, action_done
    
    def _translate(self) -> None:
        # 定义表层、次表层、深层以及环境层的平动范围
        lamada_d = 0.2
        lamada_s = 0.4
        lamada_layer = 0.6
        # lamada_env = 0

        muti_movement = np.array([np.random.normal(0.25,0.25), np.random.normal(0.25,0.25), np.random.normal(0.25,0.25)])

        initial_positions = self.atoms.positions

        for atom in initial_positions:
            if atom in self.deep_atom:
                atom += lamada_d * muti_movement
            if atom in self.sub_atom:
                atom += lamada_s * muti_movement
            if atom in self.surf_atom:
                atom += lamada_layer * muti_movement
            if atom in self.layer_atom:
                atom += lamada_layer * muti_movement
        self.atoms.positions = initial_positions
    
    def _rotate(self, atoms:Atoms, zeta:float) -> None:
        initial_state = atoms.copy()
        zeta = math.pi * zeta / 180
        central_point = np.array([initial_state.cell[0][0] / 2, initial_state.cell[1][1] / 2, 0])
        matrix = [[cos(zeta), -sin(zeta), 0],
                    [sin(zeta), cos(zeta), 0],
                    [0, 0, 1]]
        matrix = np.array(matrix)

        for atom in initial_state.positions:

            if 14.5 < atom[2] < 24.0:
                atom += np.array(
                        (np.dot(matrix, (np.array(atom.tolist()) - central_point).T).T + central_point).tolist()) - atom
        atoms.positions = initial_state.get_positions()

    @timeout_decorator.timeout(300)
    def _md(self) -> None:
        self.to_constraint(self.atoms)
        if self.calculator_method in ['MACE', 'mace', 'Mace']:
            self.atoms = self.calculator(self.atoms, calc_type = 'md')
        elif self.calculator_method in ['LASP', 'Lasp', 'lasp']:
            self.atoms = self.calculator(self.atoms, calc_type = 'ssw')

    def _md_barrier(self, previous_energy:float) -> float:
        traj = Trajectory('md.traj')
        energy_list = []
        for atoms in traj:
            energy_list.append(atoms.get_potential_energy() + 
                               self.n_O2 * self.E_O2 + 
                               self.n_O3 * self.E_O3)
        return max(energy_list) - previous_energy
    
    def _get_normal_vector(self, tri):
        A_B = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        f_vector = A_B/np.linalg.norm(A_B)
        return f_vector

    def _get_all_vector_list(self, top, bridge, hollow):
        total_n_vector = []
        for top_vector in top:
            total_n_vector.append(top_vector)
        for bridge_vector in bridge:
            total_n_vector.append(bridge_vector)
        for hollow_vector in hollow:
            total_n_vector.append(hollow_vector)
        return np.array(total_n_vector)

    def _diffuse(self, slab:Atoms):
        to_diffuse_O_list = []
        diffusable_sites = []
        diffusable_total_norm_vector = []
        interference_O_distance = []
        diffusable = True
        action_done = True

        single_layer_O_list = self.layer_O_atom_list(slab)
        total_s, total_plane_normal_vector = self.get_surf_sites(slab)
            
        for ads_index in range(len(total_s)): # 寻找可以diffuse的位点
            ads_sites = total_s[ads_index]
            ads_plane_norm_vector = total_plane_normal_vector[ads_index]

            to_other_O_distance = []
            if single_layer_O_list:
                for i in single_layer_O_list:
                    distance = self.distance(ads_sites[0] + ads_plane_norm_vector[0] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[1] + ads_plane_norm_vector[1] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[2] + ads_plane_norm_vector[2] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             slab.get_positions()[i][0],
                                             slab.get_positions()[i][1], 
                                             slab.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 1.5 * d_O_O:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                diffusable_sites.append(ads_sites)
                diffusable_total_norm_vector.append(ads_plane_norm_vector)

        if single_layer_O_list: # 防止氧原子被trap住无法diffuse
            for i in single_layer_O_list:
                to_other_O_distance = []
                for j in single_layer_O_list:
                    if j != i:
                        distance = self.distance(slab.get_positions()[i][0],
                                           slab.get_positions()[i][1], slab.get_positions()[i][2],slab.get_positions()[j][0],
                                           slab.get_positions()[j][1], slab.get_positions()[j][2])
                        to_other_O_distance.append(distance)
                        
                if self.to_get_min_distances(to_other_O_distance,4):
                    d_min_4 = self.to_get_min_distances(to_other_O_distance, 4)
                    if d_min_4 > 2.0:
                        to_diffuse_O_list.append(i)
                else:
                    to_diffuse_O_list.append(i)
        else:
            action_done = False

        if to_diffuse_O_list and action_done:
            selected_O_index = single_layer_O_list[np.random.randint(len(to_diffuse_O_list))]
            if diffusable_sites:
                diffuse_index = np.random.randint(len(diffusable_sites))
                diffuse_site = diffusable_sites[diffuse_index]
                target_diffuse_norm_vector = diffusable_total_norm_vector[diffuse_index]

            interference_O_list = [i for i in single_layer_O_list if i != selected_O_index]
            for j in interference_O_list:
                d = self.atom_to_traj_distance(slab.positions[selected_O_index], diffuse_site, slab.positions[j])
                interference_O_distance.append(d)
            if interference_O_distance:
                if min(interference_O_distance) < 0.3 * d_O_O:
                    diffusable = False
        
            if diffusable and diffuse_site[0]:
                for atom in slab:
                    if atom.index == selected_O_index:
                        atom.position = np.array([diffuse_site[0] + target_diffuse_norm_vector[0] * self.d_list[int(diffuse_site[3]) - 1], 
                                                  diffuse_site[1] + target_diffuse_norm_vector[1] * self.d_list[int(diffuse_site[3]) - 1], 
                                                  diffuse_site[2] + target_diffuse_norm_vector[2] * self.d_list[int(diffuse_site[3]) - 1]])
            else:
                action_done = False

        else:
            action_done = False

            
        return slab, action_done
    
    def _drill(self, atoms:Atoms) -> Tuple[Atoms, bool]:
        action_done = True
        selected_drill_O_list = []
        layer_O_atom_list = self.layer_O_atom_list(atoms)
        sub_O_atom_list = self.sub_O_atom_list(atoms)
        if layer_O_atom_list:
            for i in layer_O_atom_list:
                selected_drill_O_list.append(i)
        if sub_O_atom_list:
            for j in sub_O_atom_list:
                selected_drill_O_list.append(j)

        if selected_drill_O_list:
            selected_O = selected_drill_O_list[np.random.randint(len(selected_drill_O_list))]

            if selected_O in layer_O_atom_list:
                atoms = self._drill_surf(atoms)
            elif selected_O in sub_O_atom_list:
                atoms = self._drill_deep(atoms)
        else:
            action_done = False

        return atoms, action_done
    
    def get_reward_trans(self, relative_energy):
        return -relative_energy / (self.H * self.k * self.temperature)

    def get_reward_tanh(self, relative_energy):
        reward = math.tanh(-relative_energy/(self.H * self.k * self.temperature))
        return reward
    
    def get_reward_sigmoid(self, relative_energy):
        return 2 * (0.5 - 1 / (1 + np.exp(-relative_energy/(self.H * self.k * self.temperature))))
    
    def _kinetic_penalty(self, barrier: float) -> float:
        # when TOF <= 10E-5, we consider that the reaction while not occur
        # so in order to analyze whether the action can happen theorically
        # we consider that the action rate should be compared with 1/TOF
        # if k < 1 / TOF: it should get relative penalty
        # elif k >= 1 / TOF: its penalty should be zero

        # if kb = TOF = exp(Ea/RT) => Ea_min = ln(TOF) * R * T
        # we define the penalty should be penalty = exp(barrier / Ea_min) - 1
        # And the penalty should <= self.timesteps * 0.5
        if self.use_kinetic_penalty == 'non-strict':
            if barrier == 5.0:
                kinetic_penalty = max(np.exp(5.0 / (self.H * self.k * self.temperature)) - 1, 5.0)
            else:
                kinetic_penalty = np.exp(max(barrier, 0) / (self.H * self.k * self.temperature)) - 1
        elif self.use_kinetic_penalty == 'strict':
            if barrier <= np.log(self.timesteps * 0.5 + 1) * self.Ea_min:
                kinetic_penalty = np.exp(max(barrier, 0) / self.Ea_min) - 1
            else:
                kinetic_penalty = self.timesteps * 0.5
        print(f"The kinetic penalty is {kinetic_penalty}")
        return kinetic_penalty

    def lifted_distance(self, drill_site, pos):

        r = self.distance(drill_site[0], drill_site[1], drill_site[2] +1.3,
                                    pos[0], pos[1], pos[2])
        
        lifted_d = math.exp(- r * r / (2 * 2.5 ** 2))

        return min(lifted_d, 0.5)
    
    def _drill_surf(self, slab:Atoms):
        layer_O = []
        to_distance = []
        drillable_sites = []
        drillable_norm_vector = []
        layer_List = self.get_layer_list(slab)

        sub_sites, sub_normal_vector = self.get_sub_sites(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        for ads_index in range(len(sub_sites)):
            ads_sites = sub_sites[ads_index]
            ads_norm_vector = sub_normal_vector[ads_index]
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    distance = self.distance(ads_sites[0] + ads_norm_vector[0] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[1] + ads_norm_vector[1] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[2] + ads_norm_vector[2] * self.d_list[int(ads_sites[3]) - 1 ],
                                             slab.get_positions()[i][0],
                                             slab.get_positions()[i][1], 
                                             slab.get_positions()[i][2])
                    
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)
                drillable_norm_vector.append(ads_norm_vector)
        
        if layer_O:
            layer_O_atom_list = self.layer_O_atom_list(slab)

        if layer_O_atom_list:
            i = layer_O_atom_list[np.random.randint(len(layer_O_atom_list))]
            position = slab.get_positions()[i]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_index = to_distance.index(min(to_distance))
            drill_site = drillable_sites[drill_index]
            target_norm_vector = drillable_norm_vector[drill_index]
            for atom in slab:
                if atom.index == i:
                    atom.position = np.array([drill_site[0] + target_norm_vector[0] * self.d_list[int(drill_site[3]) - 1 ],
                                              drill_site[1] + target_norm_vector[1] * self.d_list[int(drill_site[3]) - 1 ],
                                              drill_site[2] + target_norm_vector[2] * self.d_list[int(drill_site[3]) - 1 ]])

            lifted_atoms_list = self.label_atoms(slab, [surf_z - 1.0, layer_z + fluct_d_layer])
            for j in lifted_atoms_list:
                slab.positions[j][2] += self.lifted_distance(drill_site, slab.get_positions()[j])
        return slab
    
    def _drill_deep(self, slab:Atoms) -> Atoms:
        to_distance = []
        drillable_sites = []
        drillable_norm_vector = []
        sub_O_atom_list = self.sub_O_atom_list(slab)
        sub_List = self.get_sub_list(slab)

        deep_sites, deep_norm_vector = self.get_deep_sites(slab)
        
        for ads_index in range(len(deep_sites)):
            ads_sites = deep_sites[ads_index]
            ads_norm_vector = deep_norm_vector[ads_index]
            to_other_O_distance = []
            if sub_List:
                for i in sub_List:
                    distance = self.distance(ads_sites[0] + ads_norm_vector[0] * self.d_list[int(ads_sites[3]) - 1 ],
                                             ads_sites[1] + ads_norm_vector[1] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             ads_sites[2] + ads_norm_vector[2] * self.d_list[int(ads_sites[3]) - 1 ], 
                                             slab.get_positions()[i][0],
                                             slab.get_positions()[i][1], 
                                             slab.get_positions()[i][2])
                    to_other_O_distance.append(distance)
                if min(to_other_O_distance) > 2 * d_O_O:
                    ads_sites[4] = 1
                else:
                    ads_sites[4] = 0
            else:
                ads_sites[4] = 1
            if ads_sites[4]:
                drillable_sites.append(ads_sites)
                drillable_norm_vector.append(ads_norm_vector)

        if sub_O_atom_list:
            i = sub_O_atom_list[np.random.randint(len(sub_O_atom_list))]
            position = slab.get_positions()[i]
            for drill_site in drillable_sites:
                to_distance.append(
                            self.distance(position[0], position[1], position[2], drill_site[0], drill_site[1],
                                        drill_site[2]))

        if to_distance:
            drill_index = to_distance.index(min(to_distance))
            drill_site = drillable_sites[drill_index]
            target_norm_vector = drillable_norm_vector[drill_index]
            for atom in slab:
                if atom.index == i:
                    atom.position = np.array([drill_site[0] + target_norm_vector[0] * self.d_list[int(drill_site[3]) - 1 ],
                                              drill_site[1] + target_norm_vector[1] * self.d_list[int(drill_site[3]) - 1 ],
                                              drill_site[2] + target_norm_vector[2] * self.d_list[int(drill_site[3]) - 1 ]])


            lifted_atoms_list = self.label_atoms(slab, [sub_z - 1.0, layer_z + fluct_d_layer])
        
            for j in lifted_atoms_list:
                slab.positions[j][2] += self.lifted_distance(drill_site, slab.get_positions()[j])
        return slab

    def _dissociate(self, slab:Atoms) -> Atoms:
        action_done = True
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O','O',unique = True)
        PdOBonds = ana.get_bonds(self.metal_ele, 'O', unique = True)

        Pd_O_list = []
        dissociate_O2_list = []

        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])

        if OOBonds[0]:
            layerList = self.get_layer_list(slab)
            for j in OOBonds[0]:
                if (j[0] in layerList or j[1] in layerList) and (j[0] in Pd_O_list or j[1] in Pd_O_list):
                    dissociate_O2_list.append([(j[0],j[1])])

        if dissociate_O2_list:
            OO = dissociate_O2_list[np.random.randint(len(dissociate_O2_list))]
            print(f"The selected O2 is {OO}")
            # print('Before rotating the atoms positions are:', slab.get_positions()[OO[0][0]], slab.get_positions()[OO[0][1]])
            slab = self.oxy_rotation(slab, OO)
            # print('Before expanding the atoms positions are:', slab.get_positions()[OO[0][0]], slab.get_positions()[OO[0][1]])
            slab = self.to_dissociate(slab, OO)
        else:
            action_done = False
        return slab, action_done

    def label_atoms(self, atoms:Atoms, zRange:List) -> List[int]:
        myPos = atoms.get_positions()
        return [
            i for i in range(len(atoms)) \
            if min(zRange) < myPos[i][2] < max(zRange)
        ]

    def distance(self, x1:float, y1:float, z1:float, x2:float, y2:float, z2:float) -> float:
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
        return dis

    def _generate_initial_slab(self) -> Atoms:
        slab = fcc100(self.metal_ele, size=(6, 6, 4), vacuum=10.0)
        delList = [77, 83, 89, 95, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 119, 120, 125,
                   126, 131, 132, 137, 138, 139, 140, 141, 142, 143]
        del slab[[i for i in range(len(slab)) if i in delList]]
        return slab
    
    def get_atom_info(self, atoms:Atoms) -> Tuple[Atoms]:
        layerList = self.get_layer_list(atoms)
        surfList = self.get_surf_list(atoms)
        subList = self.get_sub_list(atoms)
        deepList = self.get_deep_list(atoms)
        
        layer = atoms.copy()
        del layer[[i for i in range(len(layer)) if i not in layerList]]
        layer_atom = layer.get_positions()

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if i not in surfList]]
        surf_atom = surf.get_positions()

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if i not in subList]]
        sub_atom = sub.get_positions()

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if i not in deepList]]
        deep_atom = deep.get_positions()

        return layer_atom, surf_atom, sub_atom, deep_atom
    
    def RMSD(self, current_atoms, previous_atoms):
        similar = False

        constraint_p = self.get_constraint(previous_atoms).get_indices()
        constraint_c = self.get_constraint(current_atoms).get_indices()

        free_atoms_p = [atom_idx for atom_idx in range(len(previous_atoms)) if atom_idx not in constraint_p]
        free_atoms_c = [atom_idx for atom_idx in range(len(current_atoms)) if atom_idx not in constraint_c]

        RMSD = 0
        cell_x = current_atoms.cell[0][0]
        cell_y = current_atoms.cell[1][1]
        if len(free_atoms_p) == len(free_atoms_c) and len(previous_atoms) == len(current_atoms):
            for i in free_atoms_p:
                d = self.distance(previous_atoms.positions[i][0], previous_atoms.positions[i][1], previous_atoms.positions[i][2],
                                  current_atoms.positions[i][0], current_atoms.positions[i][1], current_atoms.positions[i][2])
                if d > max(cell_x, cell_y) / 2:
                    d = self._get_pbc_min_dis(previous_atoms, current_atoms, i)
                    
                RMSD += d * d
            RMSD = math.sqrt(RMSD / len(free_atoms_p))
            self.RMSD_threshold = min(len(free_atoms_p) * 0.005, 0.5)
            if RMSD <= self.RMSD_threshold:
                similar = True

        return [similar, RMSD]
    
    def get_constraint(self, atoms:Atoms) -> FixAtoms:
        bottomList = self.get_bottom_list(atoms)
        # bottomList.extend(self.get_deep_list(atoms))
        constraint = FixAtoms(mask=[a.symbol == self.metal_ele and a.index in bottomList for a in atoms])
        return constraint
    
    def to_constraint(self, atoms:Atoms) -> None:
        constraint = self.get_constraint(atoms)
        fix = atoms.set_constraint(constraint)

    def exist_too_short_bonds(self,slab:Atoms) -> bool:
        exist = False
        ana = Analysis(slab)
        PdPdBonds = ana.get_bonds(self.metal_ele,self.metal_ele,unique = True)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.metal_ele, 'O', unique=True)
        PdPdBondValues = ana.get_values(PdPdBonds)[0]
        minBonds = []
        minPdPd = min(PdPdBondValues)
        minBonds.append(minPdPd)
        if OOBonds[0]:
            OOBondValues = ana.get_values(OOBonds)[0]
            minOO = min(OOBondValues)
            minBonds.append(minOO)
        if PdOBonds[0]:    
            PdOBondValues = ana.get_values(PdOBonds)[0]
            minPdO = min(PdOBondValues)
            minBonds.append(minPdO)

        if min(minBonds) < 0.4:
            exist = True
        return exist
    
    def atom_to_traj_distance(self, atom_A:Atoms, atom_B:Atoms, atom_C:Atoms) -> float:
        d_AB = self.distance(atom_A[0], atom_A[1], atom_A[2], atom_B[0], atom_B[1], atom_B[2])
        d = abs((atom_C[0]-atom_A[0])*(atom_A[0]-atom_B[0])+
                (atom_C[1]-atom_A[1])*(atom_A[1]-atom_B[1])+
                (atom_C[2]-atom_A[2])*(atom_A[2]-atom_B[2])) / d_AB
        return d

    def get_bond_info(self, slab):
        ana = Analysis(slab)
        PdPdBonds = ana.get_bonds(self.metal_ele,self.metal_ele,unique = True)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.metal_ele, 'O', unique=True)
        PdPdBondValues = ana.get_values(PdPdBonds)[0]
        if OOBonds[0]:
            OOBondValues = ana.get_values(OOBonds)[0]
            if PdOBonds[0]:
                PdOBondValues = ana.get_values(PdOBonds)[0]
                if min(PdPdBondValues) < d_Pd_Pd * 0.80 or min(OOBondValues) < d_O_O * 0.80 or min(PdOBondValues) < d_O_Pd * 0.80 or max(OOBondValues) > 1.25 * d_O_O:
                    return True
                else:
                    return False
            else:
                if min(PdPdBondValues) < d_Pd_Pd * 0.80 or min(OOBondValues) < d_O_O * 0.80  or max(OOBondValues) > 1.25 * d_O_O:
                    return True
                else:
                    return False
        else:
            if PdOBonds[0]:
                PdOBondValues = ana.get_values(PdOBonds)[0]
                if min(PdPdBondValues) < d_Pd_Pd * 0.80 or min(PdOBondValues) < d_O_Pd * 0.80:
                    return True
                else:
                    return False
            else:
                if min(PdPdBondValues) < d_Pd_Pd * 0.80:
                    return True
                else:
                    return False
    
    def ball_func(self,pos1:np.asarray, pos2:np.asarray) -> Tuple[np.asarray, np.asarray]:	# zeta < 36, fi < 3
        d = self.distance(pos1[0],pos1[1],pos1[2],pos2[0],pos2[1],pos2[2])
        '''如果pos1[2] > pos2[2],atom_1旋转下来'''
        pos2_position = pos2

        pos_slr = pos1 - pos2

        pos_slr_square = math.sqrt(pos_slr[0] * pos_slr[0] + pos_slr[1] * pos_slr[1])

        if pos_slr_square:
            pos1_position = [pos2[0] + d * pos_slr[0]/pos_slr_square, pos2[1] + d * pos_slr[1]/pos_slr_square, pos2[2]]
        else:
            pos1_position = pos1
        
        return pos1_position, pos2_position

    def oxy_rotation(self, slab:Atoms, OO:List[int]) -> Atoms:
        if slab.positions[OO[0][0]][2] > slab.positions[OO[0][1]][2]:
            a,b = self.ball_func(slab.get_positions()[OO[0][0]], slab.get_positions()[OO[0][1]])
        else:
            a,b = self.ball_func(slab.get_positions()[OO[0][1]], slab.get_positions()[OO[0][0]])
        slab.positions[OO[0][0]] = a
        slab.positions[OO[0][1]] = b
        return slab
    
    def to_dissociate(self, slab:Atoms, atoms:List[int]) -> Atoms:
        expanding_index = 2.0
        central_point = np.array([(slab.get_positions()[atoms[0][0]][0] + slab.get_positions()[atoms[0][1]][0])/2, 
                                  (slab.get_positions()[atoms[0][0]][1] + slab.get_positions()[atoms[0][1]][1])/2, (slab.get_positions()[atoms[0][0]][2] + slab.get_positions()[atoms[0][1]][2])/2])
        
        slab.positions[atoms[0][0]] += np.array([expanding_index*(slab.get_positions()[atoms[0][0]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][0]][2]-central_point[2])])
        
        slab.positions[atoms[0][1]] += np.array([expanding_index*(slab.get_positions()[atoms[0][1]][0]-central_point[0]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][1]-central_point[1]), 
                                                 expanding_index*(slab.get_positions()[atoms[0][1]][2]-central_point[2])])
        
        # print('after expanding, the positions of the atoms are', slab.get_positions()[atoms[0][0]], slab.get_positions()[atoms[0][1]])
        addable_sites = []
        addable_total_norm_vector = []
        layer_O = []
        layerlist = self.get_layer_list(slab)

        total_s, total_plane_normal_vector = self.get_surf_sites(slab)

        for ads_index in range(len(total_s)):
            ads_site = total_s[ads_index]
            ads_plane_norm_vector = total_plane_normal_vector[ads_index]

            for i in layerlist:
                if slab[i].symbol == 'O':
                    layer_O.append(i)
            to_other_O_distance = []
            if layer_O:
                for i in layer_O:
                    to_distance = self.distance(ads_site[0] + ads_plane_norm_vector[0] * self.d_list[int(ads_site[3]) - 1 ], 
                                             ads_site[1] + ads_plane_norm_vector[1] * self.d_list[int(ads_site[3]) - 1 ], 
                                             ads_site[2] + ads_plane_norm_vector[2] * self.d_list[int(ads_site[3]) - 1 ], 
                                             slab.get_positions()[i][0],
                                             slab.get_positions()[i][1], 
                                             slab.get_positions()[i][2])
                    to_other_O_distance.append(to_distance)
                if min(to_other_O_distance) > 1.5 * d_O_O:
                    ads_site[4] = 1
            else:
                ads_site[4] = 1
            if ads_site[4]:
                addable_sites.append(ads_site)
                addable_total_norm_vector.append(ads_plane_norm_vector)

        if addable_sites:
            print("The num of addable sites is:", len(addable_sites))
            O1_distance = []
            for add_1_index in range(len(addable_sites)):
                add_1_site = addable_sites[add_1_index]
                add_1_norm_vector = addable_total_norm_vector[add_1_index]
                distance_1 = self.distance(add_1_site[0] + add_1_norm_vector[0] * self.d_list[int(add_1_site[3]) - 1 ], 
                                           add_1_site[1] + add_1_norm_vector[1] * self.d_list[int(add_1_site[3]) - 1 ], 
                                           add_1_site[2] + add_1_norm_vector[2] * self.d_list[int(add_1_site[3]) - 1 ], 
                                           slab.get_positions()[atoms[0][0]][0],
                                           slab.get_positions()[atoms[0][0]][1], 
                                           slab.get_positions()[atoms[0][0]][2])
                O1_distance.append(distance_1)

            O1_site = addable_sites[O1_distance.index(min(O1_distance))]
            O1_norm_vector = addable_total_norm_vector[O1_distance.index(min(O1_distance))]
            
            ad_2_sites, ad_2_norm_vector = [], []
            for add_index in range(len(addable_sites)):
                add_site = addable_sites[add_index]
                add_norm_vector = addable_total_norm_vector[add_index]
                d = self.distance(add_site[0] + add_norm_vector[0] * self.d_list[int(add_site[3]) - 1 ], 
                                  add_site[1] + add_norm_vector[1] * self.d_list[int(add_site[3]) - 1 ], 
                                  add_site[2] + add_norm_vector[2] * self.d_list[int(add_site[3]) - 1 ], 
                                  O1_site[0], O1_site[1], O1_site[2])
                if d > 2.0 * d_O_O:
                    ad_2_sites.append(add_site)
                    ad_2_norm_vector.append(add_norm_vector)

            O2_distance = []
            for add_2_index in range(len(ad_2_sites)):
                add_2_site = addable_sites[add_2_index]
                add_2_norm_vector = addable_total_norm_vector[add_2_index]
                distance_2 = self.distance(add_2_site[0] + add_2_norm_vector[0] * self.d_list[int(add_2_site[3]) - 1 ], 
                                           add_2_site[1] + add_2_norm_vector[1] * self.d_list[int(add_2_site[3]) - 1 ], 
                                           add_2_site[2] + add_2_norm_vector[2] * self.d_list[int(add_2_site[3]) - 1 ], 
                                           slab.get_positions()[atoms[0][1]][0],
                                           slab.get_positions()[atoms[0][1]][1], 
                                           slab.get_positions()[atoms[0][1]][2])
                O2_distance.append(distance_2)
            
            if O2_distance:
                O2_site = ad_2_sites[O2_distance.index(min(O2_distance))]
                O2_norm_vector = addable_total_norm_vector[O2_distance.index(min(O2_distance))]
            else:
                O2_site = O1_site
                O2_norm_vector = O1_norm_vector

            for atom in slab:
                if O1_site[0] == O2_site[0] and O1_site[1] == O2_site[1]:
                    O_1_position = np.array([O1_site[0] + O1_norm_vector[0] * self.d_list[int(O1_site[3]) - 1 ],
                                             O1_site[1] + O1_norm_vector[1] * self.d_list[int(O1_site[3]) - 1 ],
                                             O1_site[2] + O1_norm_vector[2] * self.d_list[int(O1_site[3]) - 1 ]])
                    
                    O_2_position = np.array([O1_site[0] + O1_norm_vector[0] * self.d_list[int(O1_site[3]) - 1 ],
                                             O1_site[1] + O1_norm_vector[1] * self.d_list[int(O1_site[3]) - 1 ],
                                             O1_site[2] + O1_norm_vector[2] * self.d_list[int(O1_site[3]) - 1 ] + 1.21])
                else:
                    O_1_position = np.array([O1_site[0] + O1_norm_vector[0] * self.d_list[int(O1_site[3]) - 1 ],
                                             O1_site[1] + O1_norm_vector[1] * self.d_list[int(O1_site[3]) - 1 ],
                                             O1_site[2] + O1_norm_vector[2] * self.d_list[int(O1_site[3]) - 1 ]])
                    
                    O_2_position = np.array([O2_site[0] + O2_norm_vector[0] * self.d_list[int(O2_site[3]) - 1 ],
                                             O2_site[1] + O2_norm_vector[1] * self.d_list[int(O2_site[3]) - 1 ],
                                             O2_site[2] + O2_norm_vector[2] * self.d_list[int(O2_site[3]) - 1 ]])

                if atom.index == atoms[0][0]:
                        atom.position = O_1_position
                elif atom.index == atoms[0][1]:
                    atom.position = O_2_position
            
            print('O1 position is:', O_1_position)
            print('O2 position is:', O_2_position)

        return slab
    
    def get_angle_with_z(self,slab, atoms):
        if slab.positions[atoms[0][0]][2] > slab.positions[atoms[0][1]][2]:
            a = np.array([slab.get_positions()[atoms[0][0]][0] - slab.get_positions()[atoms[0][1]][0], slab.get_positions()[atoms[0][0]][1] - slab.get_positions()[atoms[0][1]][1], slab.get_positions()[atoms[0][0]][2] - slab.get_positions()[atoms[0][1]][2]])
        else:
            a = np.array([slab.get_positions()[atoms[0][1]][0] - slab.get_positions()[atoms[0][0]][0], slab.get_positions()[atoms[0][1]][1] - slab.get_positions()[atoms[0][0]][1], slab.get_positions()[atoms[0][1]][2] - slab.get_positions()[atoms[0][0]][2]])
        z = np.array([0,0,1])
        zeta = math.asin(np.dot(a,z)/math.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]))
        return zeta
    
    def _get_pbc_min_dis(self, atoms_1, atoms_2, i):
        atom_x = atoms_1.cell[0][0]
        atom_y = atoms_1.cell[1][1]

        d = []
        atom_1_x = atoms_1.get_positions()[i][0]
        atom_1_y = atoms_1.get_positions()[i][1]
        atom_1_z = atoms_1.get_positions()[i][2]

        atom_2_x = [atoms_2.get_positions()[i][0], atoms_2.get_positions()[i][0] - atom_x, atoms_2.get_positions()[i][0] + atom_x]
        atom_2_y = [atoms_2.get_positions()[i][1], atoms_2.get_positions()[i][1] - atom_y, atoms_2.get_positions()[i][1] + atom_y]
        z = atoms_2.get_positions()[i][2]

        for x in atom_2_x:
            for y in atom_2_y:
                d.append(self.distance(atom_1_x, atom_1_y, atom_1_z, x, y, z))
        
        dis = min(d)
        return dis
    
    def to_get_min_distances(self, a, min_point):
        for i in range(len(a) - 1):
            for j in range(len(a)-i-1):
                if a[j] > a[j+1]:
                    a[j], a[j+1] = a[j+1], a[j]
        if len(a):
            if len(a) < min_point:
                return a[-1]
            else:
                return a[min_point - 1]
        else:
            return False

    def RMSE(self, a:List) -> float:
        mean = np.mean(a)
        b = mean * np.ones(len(a))
        diff = np.subtract(a, b)
        square = np.square(diff)
        MSE = square.mean()
        RMSE = np.sqrt(MSE)
        return RMSE
    
    def del_env_adsorbate(self, slab:Atoms) -> Atoms:
        slab, del_O3 = self.del_env_O3(slab)
        slab, del_O2 = self.del_env_O2(slab)

        return slab, bool(del_O2 or del_O3)
    
    def _env_O(self, slab:Atoms) -> Atoms:
        ana = Analysis(slab)

        Bonded_O_list = []
        env_O_list = []
        for atom_symbol in [self.metal_ele, 'Si', 'O', 'H']:
            EleOBonds = ana.get_bonds(atom_symbol, 'O', unique=True)
            if EleOBonds[0]:
                for EleO_bond in EleOBonds[0]:
                    Bonded_O_list.extend([EleO_bond[0], EleO_bond[1]])

        env_O_list = [atom_idx for atom_idx in range(len(slab)) 
                    if slab[atom_idx].symbol == 'O' and atom_idx not in Bonded_O_list]
        if env_O_list:
            print(f"current del env O list is {env_O_list}")
        del slab[[atom_idx for atom_idx in range(len(slab)) if atom_idx in env_O_list]]

        return env_O_list

    def del_env_O2(self, slab:Atoms) -> Atoms:
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.metal_ele, 'O', unique=True)

        Pd_O_list = []
        del_list = []

        if PdOBonds[0]:
            for PdO_bond in PdOBonds[0]:
                Pd_O_list.extend([PdO_bond[0], PdO_bond[1]])
        
        if OOBonds[0]:
            for OO in OOBonds[0]:
                if OO[0] not in Pd_O_list and OO[1] not in Pd_O_list:
                    del_list.extend([OO[0], OO[1]])
                    self.ads_list.append(2)

        del_list = [del_idx for n,del_idx in enumerate(del_list) if del_idx not in del_list[:n]]
        if del_list:
            print(f"current del env O2 list is {del_list}")
        del slab[[atom_idx for atom_idx in range(len(slab)) if atom_idx in del_list]]

        self.n_O2 += int(len(del_list)/2)

        return slab, bool(del_list)


    def del_env_O3(self, slab:Atoms) -> Atoms:
        ana = Analysis(slab)
        PdOBonds = ana.get_bonds(self.metal_ele, 'O', unique=True)
        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        Pd_O_list = []
        del_list = []
        if PdOBonds[0]:
            for PdO_bond in PdOBonds[0]:
                Pd_O_list.extend([PdO_bond[0], PdO_bond[1]])

        if OOOangles[0]:
            for O3 in OOOangles[0]:
                if O3[0] not in Pd_O_list and O3[1] not in Pd_O_list and O3[2] not in Pd_O_list:
                    del_list.extend([O3[0], O3[1], O3[2]])
                    self.ads_list.append(3)

        del_list = [del_idx for n,del_idx in enumerate(del_list) if del_idx not in del_list[:n]]
        if del_list:
            print(f"current del env O3 list is {del_list}")

        # To check whether the OOOO exists
        if np.ceil(len(del_list)/3) == int(len(del_list)/3):
            del slab[[atom_idx for atom_idx in range(len(slab)) if atom_idx in del_list]]
            self.n_O3 += int(len(del_list)/3)
        else:
            write_arc([slab], name = "broken_slab.arc")
        
        return slab, bool(del_list)
    
    def to_desorb_adsorbate(self, slab:Atoms):
        desorb = ()
        ana = Analysis(slab)
        OOBonds = ana.get_bonds('O', 'O', unique = True)
        PdOBonds = ana.get_bonds(self.metal_ele, 'O', unique=True)

        OOOangles = ana.get_angles('O', 'O', 'O',unique = True)

        Pd_O_list = []
        desorb_list = []
        if PdOBonds[0]:
            for i in PdOBonds[0]:
                Pd_O_list.append(i[0])
                Pd_O_list.append(i[1])
        
        if OOBonds[0]:  # 定义环境中的氧气分子
            for i in OOBonds[0]:
                if i[0] in Pd_O_list or i[1] in Pd_O_list:
                    desorb_list.append(i)

        if OOOangles[0]:
            for j in OOOangles[0]:
                if j[0] in Pd_O_list or j[1] in Pd_O_list or j[2] in Pd_O_list:
                    desorb_list.append(j)

        if desorb_list:
            desorb = desorb_list[np.random.randint(len(desorb_list))]
        return desorb
    
    def _2D_distance(self, x1:float,x2:float, y1:float,y2:float) -> float:
        dis = math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
        return dis
    
    def get_layer_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [layer_z - fluct_d_layer, layer_z + fluct_d_layer])
    
    def get_surf_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [surf_z - fluct_d_metal, surf_z + fluct_d_metal*2])
    
    def get_sub_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [sub_z - fluct_d_metal, sub_z + fluct_d_metal])
    
    def get_deep_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [deep_z - fluct_d_metal, deep_z + fluct_d_metal])
    
    def get_bottom_list(self, atoms:Atoms) -> List[int]:
        return self.label_atoms(atoms, [9.0, 13.0])

    def layer_O_atom_list(self, slab:Atoms) -> List[int]:
        layer_O = []
        layer_O_atom_list = []
        layer_OObond_list = []
        layer_List = self.get_layer_list(slab)

        for i in slab:
            if i.index in layer_List and i.symbol == 'O':
                layer_O.append(i.index)
        
        if layer_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in layer_O or i[1] in layer_O:
                        layer_OObond_list.append(i[0])
                        layer_OObond_list.append(i[1])

            for j in layer_O:
                if j not in layer_OObond_list:
                    layer_O_atom_list.append(j)
        return layer_O_atom_list
    
    def sub_O_atom_list(self, slab:Atoms) -> List[int]:
        sub_O = []
        sub_O_atom_list = []
        sub_OObond_list = []
        sub_List = self.get_sub_list(slab)

        for i in slab:
            if i.index in sub_List and i.symbol == 'O':
                sub_O.append(i.index)
        
        if sub_O:
            ana = Analysis(slab)
            OObonds = ana.get_bonds('O','O',unique = True)
            if OObonds[0]:
                for i in OObonds[0]:
                    if i[0] in sub_O and i[1] in sub_O:
                        sub_OObond_list.append(i[0])
                        sub_OObond_list.append(i[1])

            for j in sub_O:
                if j not in sub_OObond_list:
                    sub_O_atom_list.append(j)
        return sub_O_atom_list

    # get layer's sites
    def get_surf_sites(self, atoms:Atoms) -> np.asarray:
        surfList = self.get_surf_list(atoms)

        surf = atoms.copy()
        del surf[[i for i in range(len(surf)) if (i not in surfList) or surf[i].symbol != self.metal_ele]]
        
        surf_sites, site_vector = self.get_sites(surf)

        return surf_sites, site_vector
    
    def get_sub_sites(self, atoms:Atoms) -> np.asarray:
        subList = self.get_sub_list(atoms)

        sub = atoms.copy()
        del sub[[i for i in range(len(sub)) if (i not in subList) or sub[i].symbol != self.metal_ele]]

        sub_sites, site_vector = self.get_sites(sub)
        return sub_sites, site_vector
    
    def get_deep_sites(self, atoms:Atoms) -> np.asarray:
        deepList = self.get_deep_list(atoms)

        deep = atoms.copy()
        del deep[[i for i in range(len(deep)) if (i not in deepList) or deep[i].symbol != self.metal_ele]]

        deep_sites, site_vector = self.get_sites(deep)

        return deep_sites, site_vector
    
    def get_sites(self, atoms:Atoms) -> np.asarray:
        if len(atoms) == 0:
            return None, None
        elif len(atoms) == 1:
            sites = []
            for _ in range(2):
                sites.append(np.array([atoms.get_positions()[0][0],atoms.get_positions()[0][1],atoms.get_positions()[0][2], 1, 0]))
            return np.array(sites), np.array([0,0,1])
        elif len(atoms) == 2:
            sites = []
            total_vectors = []
            for atom in atoms:
                sites.append(np.append(atom.position, [1, 0]))
                total_vectors.append([0,0,1])
            sites.append(np.array([(atoms.get_positions()[0][0] + atoms.get_positions()[1][0]) / 2,
                                   (atoms.get_positions()[0][1] + atoms.get_positions()[1][1]) / 2,
                                   (atoms.get_positions()[0][2] + atoms.get_positions()[1][2]) / 2,
                                   2, 0]))
            total_vectors.append([0,0,1])
            total_vectors = np.array(total_vectors)
            return np.array(sites), total_vectors
        elif len(atoms) >= 3:
            atop = atoms.get_positions()
            pos_ext = atoms.get_positions()
            tri = Delaunay(pos_ext[:, :2])
            pos_nodes = pos_ext[tri.simplices]

            bridge_sites = []
            hollow_sites = []

            top_vector, bridge_vector, hollow_vector = [], [], []
            for _ in range(len(atop)):
                top_vector.append([0, 0, 1])

            for i in pos_nodes:
                if (self.distance(i[0][0], i[0][1], i[0][2], i[1][0], i[1][1], i[1][2])) < 3.0:
                    bridge_sites.append((i[0] + i[1]) / 2)
                    bridge_vector.append(self._get_normal_vector(i))
                else:
                    hollow_sites.append((i[0] + i[1]) / 2)
                    hollow_vector.append(self._get_normal_vector(i))
                if (self.distance(i[2][0], i[2][1], i[2][2], i[1][0], i[1][1], i[1][2])) < 3.0:
                    bridge_sites.append((i[2] + i[1]) / 2)
                    bridge_vector.append(self._get_normal_vector(i))
                else:
                    hollow_sites.append((i[2] + i[1]) / 2)
                    hollow_vector.append(self._get_normal_vector(i))
                if (self.distance(i[0][0], i[0][1], i[0][2], i[2][0], i[2][1], i[2][2])) < 3.0:
                    bridge_sites.append((i[0] + i[2]) / 2)
                    bridge_vector.append(self._get_normal_vector(i))
                else:
                    hollow_sites.append((i[0] + i[2]) / 2)
                    hollow_vector.append(self._get_normal_vector(i))

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
            total_plane_vector = self._get_all_vector_list(top_vector, bridge_vector, hollow_vector)

        return total_sites, total_plane_vector

    def add_mole(self, mole:str) -> float:
        mole = molecule(mole)
        energy = self.calculator(mole, calc_type="single")
        return energy
    
    def _use_MLP(self, atoms:Atoms) -> List:
        nodes_scalar = torch.tensor(to_pad_the_array(atoms.get_atomic_numbers(), 
                                    self.max_observation_atoms, position = False, symbol = True))
        atom_embeddings = nn.Embedding(min(118,max(len(atoms), 50)),50)
        nodes_scalar = atom_embeddings(nodes_scalar)

        pos = self.pd(torch.tensor(atoms.get_positions()))
        pos = rearrange(pos.unsqueeze(1), 'a b c ->a c b')

        # print(torch.mul(nodes_scalar.unsqueeze(1), pos).detach().numpy().shape)
        return torch.mul(nodes_scalar.unsqueeze(1), pos).detach().numpy()
    
    def _use_Painn_description(self, atoms:Atoms) -> List:
        input_dict = Painn.atoms_to_graph_dict(atoms, self.cutoff)
        atom_model = Painn.PainnDensityModel(
            num_interactions = self.num_interactions,
            hidden_state_size = self.hidden_state_size,
            cutoff = self.cutoff,
            atoms = atoms,
            embedding_size = self.embedding_size,
        )

        torch.set_default_dtype(torch.float32)
        atom_representation_scalar, atom_representation_vector = atom_model(input_dict)
        
        atom_representation_scalar = np.array(self.pd(torch.tensor(np.array(atom_representation_scalar[0].tolist()))))

        # print(atom_representation_vector[0].shape)
        atom_representation_vector = rearrange(atom_representation_vector[0], "a b c -> b a c")
        # print(atom_representation_vector.shape)

        atom_representation_vector = np.array(self.pd(torch.tensor(np.array(atom_representation_vector.tolist()))))
        # print(atom_representation_vector.shape)
        atom_representation_vector = rearrange(atom_representation_vector, "b a c -> a b c")
        # print(atom_representation_vector.shape)

        return [atom_representation_scalar, atom_representation_vector]
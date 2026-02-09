import gymnasium as gym 
from gymnasium import spaces
import numpy as np 



class FJSSPEnv(gym.Env):
    def __init__(self, duration_param_args: dict=None):
        super(FJSSPEnv, self).__init__()
        
        # 1. Digital Twin Configuration: CPPUs (Machines)
        self.machines = [f'cppu{i}' for i in range(10)]
        self.capability_map = {
            'configure': ['cppu0'], 
            'label':     ['cppu1', 'cppu2', 'cppu3'],
            'box':       ['cppu4', 'cppu5', 'cppu6'], 
            'mill':      ['cppu0', 'cppu1', 'cppu3'],
            'heating':   ['cppu7', 'cppu3', 'cppu6'], 
            'inspect':   ['cppu8', 'cppu9', 'cppu0'],
            'sintering': ['cppu0'], 
            'coating':   ['cppu8', 'cppu6'], 
            'store':     ['cppu2', 'cppu5', 'cppu9']
        }

        self.duration_params = {
            ('cppu0', 'configure'): (3.214, 0.1), ('cppu1', 'label'): (1.471, 0.527),
            ('cppu3', 'mill'): (3.461, 0.1), ('cppu7', 'heating'): (-1.648, 1.048),
            ('cppu6', 'coating'): (0.688, 0.1), ('cppu9', 'inspect'): (0.353, 0.325),
            ('cppu5', 'box'): (2.192, 0.1), ('cppu2', 'store'): (1.451, 0.889),
            'default': (1.0, 0.2)
        }
        if duration_param_args: self.duration_params.update(duration_param_args)

        self.job_routings = {
            'power_supply': ['configure', 'label', 'box', 'store'],
            'brake': ['mill', 'heating', 'inspect', 'box', 'label', 'store'],
            'gear': ['sintering', 'inspect', 'heating', 'coating', 'inspect', 'box', 'store']
        }

        # 2. Spaces
        self.total_ops = 17 
        self.observation_space = spaces.Dict({
            "op_nodes": spaces.Box(low=0, high=1e5, shape=(self.total_ops, 4), dtype=np.float32), 
            "mc_nodes": spaces.Box(low=0, high=1e5, shape=(10, 2), dtype=np.float32),
            "edge_index_om": spaces.Box(low=-1, high=self.total_ops, shape=(2, 60), dtype=np.int32)
        })
        self.action_space = spaces.Discrete(41) 

    def _get_info(self):
        """Creates the metadata dictionary for the Actor-Critic indexing."""
        mapping = {}
        for j_idx, job in enumerate(self.jobs):
            if not job['finished']:
                mapping[j_idx] = job['ops_indices'][job['step']]
        
        return {
            "job_to_op_map": mapping,
            "compatible_pairs": self._get_compatible_pairs()
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.mc_available_at = {m: 0.0 for m in self.machines}
        
        self.jobs = []
        self.op_list = [] 
        for j_idx, (j_type, skills) in enumerate(self.job_routings.items()):
            job_dict = {
                'type': j_type, 'step': 0, 'finished': False, 
                'ready_at': 0.0, 'ops_indices': [], 'skills': skills
            }
            for skill in skills:
                op_id = len(self.op_list)
                self.op_list.append({
                    'job_idx': j_idx, 'skill': skill, 'finished': False, 'started': False
                })
                job_dict['ops_indices'].append(op_id)
            self.jobs.append(job_dict)
            
        return self._get_obs(), self._get_info()

    def step(self, action_idx):
        all_pairs = self._get_compatible_pairs()
        reward = 0.0
        
        # Action Logic: Attempt to start a job
        if action_idx > 0 and (action_idx - 1) < len(all_pairs):
            job_idx, mc_id = all_pairs[action_idx - 1]
            job = self.jobs[job_idx]
            op = self.op_list[job['ops_indices'][job['step']]]

            if job['ready_at'] <= self.current_time and self.mc_available_at[mc_id] <= self.current_time:
                mu, sigma = self.duration_params.get((mc_id, op['skill']), self.duration_params['default'])
                duration = np.random.lognormal(mu, sigma)
                
                finish_time = self.current_time + duration
                self.mc_available_at[mc_id] = finish_time
                op['started'], op['finished'] = True, True
                job['step'] += 1
                job['ready_at'] = finish_time
                
                if job['step'] >= len(job['ops_indices']):
                    job['finished'] = True
                    reward += 10.0
            else:
                reward -= 1.0 

        # Time Auto-Tick
        self.current_time += 1.0
        reward -= 0.1 
        
        return self._get_obs(), reward, self._is_done(), False, self._get_info()

    def _get_obs(self):
        op_feats = np.zeros((self.total_ops, 4), dtype=np.float32)
        for i, op in enumerate(self.op_list):
            job = self.jobs[op['job_idx']]
            status = [1, 1] if op['finished'] else ([1, 0] if op['started'] else [0, 0])
            progress = job['step'] / len(job['ops_indices'])
            ready_in = max(0.0, job['ready_at'] - self.current_time)
            op_feats[i] = status + [ready_in, progress]

        mc_feats = np.zeros((10, 2), dtype=np.float32)
        for i, m in enumerate(self.machines):
            free_in = max(0.0, self.mc_available_at[m] - self.current_time)
            cap_count = len([s for s, ml in self.capability_map.items() if m in ml])
            mc_feats[i] = [free_in, cap_count]

        edge_index = np.full((2, 60), -1, dtype=np.int32)
        edges = [[], []]
        for i, op in enumerate(self.op_list):
            if not op['finished'] and not op['started']:
                for m_name in self.capability_map.get(op['skill'], []):
                    edges[0].append(i)
                    edges[1].append(int(m_name.replace('cppu', '')))
        
        if edges[0]:
            num_e = min(len(edges[0]), 60)
            edge_index[0, :num_e], edge_index[1, :num_e] = edges[0][:num_e], edges[1][:num_e]

        return {"op_nodes": op_feats, "mc_nodes": mc_feats, "edge_index_om": edge_index}

    def _get_compatible_pairs(self):
        actions = []
        for i, job in enumerate(self.jobs):
            if not job['finished']:
                op = self.op_list[job['ops_indices'][job['step']]]
                for m in self.capability_map.get(op['skill'], []):
                    actions.append((i, m))
        return actions

    def _is_done(self):
        return all(j['finished'] for j in self.jobs)
    



def get_fjssp_env(duration_param_args: dict=None):
    gym.register('FJSSP-v0', entry_point=FJSSPEnv)
    return gym.make("FJSSP-v0", duration_param_args)


import gymnasium as gym 
from gymnasium import spaces
import numpy as np 



class FJSSPEnv(gym.Env):
    def __init__(self, duration_param_args=None):
        super(FJSSPEnv, self).__init__()
        
        # machines (cppus)
        self.machines = [f'cppu{i}' for i in range(10)]
        self.capability_map = {
            'configure': ['cppu0'], 
            'label': ['cppu1', 'cppu2', 'cppu3'],
            'box': ['cppu4', 'cppu5', 'cppu6'], 
            'mill': ['cppu0', 'cppu1', 'cppu3'],
            'heating': ['cppu7', 'cppu3', 'cppu6'], 
            'inspect': ['cppu8', 'cppu9', 'cppu0'],
            'sintering': ['cppu0'], 
            'coating': ['cppu8', 'cppu6'], 
            'store': ['cppu2', 'cppu5', 'cppu9']
        }

        self.total_ops = 17 
        self.observation_space = spaces.Dict({
            "op_nodes": spaces.Box(low=0, high=1e5, shape=(self.total_ops, 4), dtype=np.float32), 
            "mc_nodes": spaces.Box(low=0, high=1e5, shape=(10, 2), dtype=np.float32),
            "edge_index_om": spaces.Box(low=-1, high=self.total_ops, shape=(2, 60), dtype=np.int32)
        })
        # Action 0 = Wait, Actions 1-N = Start a specific (Job, Machine) pair
        self.action_space = spaces.Discrete(41) 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        self.mc_available_at = {m: 0.0 for m in self.machines}
        
        self.jobs = []
        self.op_list = [] 
        for j_idx, (j_type, skills) in enumerate(self.job_routings.items()):
            job_dict = {'type': j_type, 'step': 0, 'finished': False, 'ready_at': 0.0, 'ops_indices': [], 'skills': skills}
            for skill in skills:
                op_id = len(self.op_list)
                self.op_list.append({'job_idx': j_idx, 'skill': skill, 'finished': False, 'started': False})
                job_dict['ops_indices'].append(op_id)
            self.jobs.append(job_dict)
            
        return self._get_obs(), {}

    def step(self, action_idx):
        """
        Decision occurs at current_time. 
        After the decision is made, time AUTOMATICALLY increases by 1.0.
        """
        # Get all COMPATIBLE pairs (ignores if they are ready right now)
        all_pairs = self._get_compatible_pairs()
        
        reward = 0.0
        
        # Action Logic
        # If action_idx > 0 and exists in our pairs list
        if action_idx > 0 and (action_idx - 1) < len(all_pairs):
            job_idx, mc_id = all_pairs[action_idx - 1]
            job = self.jobs[job_idx]
            op = self.op_list[job['ops_indices'][job['step']]]

            # CAN ONLY START if both machine and job are ready AT THIS MOMENT
            if job['ready_at'] <= self.current_time and self.mc_available_at[mc_id] <= self.current_time:
                # Calculate duration
                mu, sigma = self.duration_params.get((mc_id, op['skill']), (1.0, 0.2))
                duration = np.random.lognormal(mu, sigma)
                
                # Update status
                finish_time = self.current_time + duration
                self.mc_available_at[mc_id] = finish_time
                op['started'], op['finished'] = True, True
                job['step'] += 1
                job['ready_at'] = finish_time
                
                if job['step'] >= len(job['ops_indices']):
                    job['finished'] = True
                    reward += 10.0 # Completion bonus
            else:
                # Invalid start attempt (machine busy or job not ready)
                reward -= 2.0 

        # --- THE AUTOMATIC TICK ---
        self.current_time += 1.0
        
        # Constant time penalty to minimize makespan
        reward -= 0.1 
        
        return self._get_obs(), reward, self._is_done(), False, {}

    def _get_obs(self):
        # Nodes and edges are calculated based on the NEW current_time
        op_feats = np.zeros((self.total_ops, 4), dtype=np.float32)
        for i, op in enumerate(self.op_list):
            job = self.jobs[op['job_idx']]
            status = [1, 1] if op['finished'] else ([0, 1] if op['started'] else [1, 0])
            progress = job['step'] / len(job['ops_indices'])
            # Feature: How long until this job is ready? (0 if already ready)
            time_to_ready = max(0.0, job['ready_at'] - self.current_time)
            op_feats[i] = status + [time_to_ready, progress]

        mc_feats = np.zeros((10, 2), dtype=np.float32)
        for i, m in enumerate(self.machines):
            # Feature: How long until this machine is free? (0 if already free)
            time_to_idle = max(0.0, self.mc_available_at[m] - self.current_time)
            cap_count = len([s for s, ml in self.capability_map.items() if m in ml])
            mc_feats[i] = [time_to_idle, cap_count]

        # Edges for GNN: Connect compatible pairs for unfinished/unstarted tasks
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
        # Logic for Actor: What combinations are physically possible?
        actions = []
        for i, job in enumerate(self.jobs):
            if not job['finished']:
                op = self.op_list[job['ops_indices'][job['step']]]
                for m in self.capability_map.get(op['skill'], []):
                    actions.append((i, m))
        return actions

    def _is_done(self):
        return all(j['finished'] for j in self.jobs)
    


gym.register('FJSSP-v0', entry_point=FJSSPEnv)
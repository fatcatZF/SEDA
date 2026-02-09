import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


from policy import ActorCritic
from fjssp_env import get_fjssp_env

import numpy as np 


# Hyperparameters
GAMMA = 0.99
K_EPOCHS = 5
EPS_CLIP = 0.2
LR = 3e-4
UPDATE_TIMESTEP = 400  # Update every 1000 environment steps
EVAL_INTERVAL = 2     # Evaluate every 2 updates

def train():
    env = get_fjssp_env()
    # op_dim=4, mc_dim=2, embed_dim=64
    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    memory = []
    best_eval_reward = -float('inf')
    timestep = 0
    updates = 0

    for episode in range(1, 5000):
        obs, info = env.reset()
        episode_reward = 0
        
        while True:
            timestep += 1
            
            # Prepare Tensors from observation
            op_t = torch.tensor(obs['op_nodes'], dtype=torch.float32)
            mc_t = torch.tensor(obs['mc_nodes'], dtype=torch.float32)
            raw_edges = obs['edge_index_om']
            mask = raw_edges[0] != -1
            edge_t = torch.tensor(raw_edges[:, mask], dtype=torch.long)

            # Model Forward Pass
            # Updated to handle your new return: v_s, all_logits, h_op, h_mc
            v_s, logits, _, _ = model(op_t, mc_t, edge_t, info)
            
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            
            # Env Step
            next_obs, reward, done, _, next_info = env.step(action.item())
            
            # Store in Memory (Keeping only what's needed for the PPO update)
            memory.append({
                'op_t': op_t, 'mc_t': mc_t, 'edge_t': edge_t, 'info': info,
                'action': action, 'log_prob': dist.log_prob(action),
                'reward': reward, 'done': done, 'v_s': v_s
            })
            
            obs, info = next_obs, next_info
            episode_reward += reward

            # Trigger Update
            if timestep % UPDATE_TIMESTEP == 0:
                updates += 1
                update_policy(model, optimizer, memory)
                memory = []
                
                # Best Policy Check
                if updates % EVAL_INTERVAL == 0:
                    stats = evaluate_policy(env, model)
                    # Comprehensive logging
                    print("-" * 30)
                    print(f"Update: {updates} | Reward Mean: {stats['reward_mean']:.2f} (±{stats['reward_std']:.2f})")
                    print(f"Avg Makespan: {stats['makespan_mean']:.1f} time units")
                    print(f"Product Completion Rates:")
                    print(f"  - Power Supply: {stats['avg_finish_power_supply']:.1f}%")
                    print(f"  - Brake:        {stats['avg_finish_brake']:.1f}%")
                    print(f"  - Gear:         {stats['avg_finish_gear']:.1f}%")
                    print("-" * 30)
    
                    # Save based on mean reward
                    if stats['reward_mean'] > best_eval_reward:
                        best_eval_reward = stats['reward_mean']
                        torch.save(model.state_dict(), "./checkpoints/best_fjssp_policy.pt")
                        print("best policy so far!")

            if done:
                break

def update_policy(model, optimizer, memory):
    # 1. Compute discounted returns
    rewards = []
    discounted_reward = 0
    for m in reversed(memory):
        if m['done']: discounted_reward = 0
        discounted_reward = m['reward'] + (GAMMA * discounted_reward)
        rewards.insert(0, discounted_reward)
    
    returns = torch.tensor(rewards, dtype=torch.float32).view(-1, 1)
    
    # 2. Optimization loop
    for _ in range(K_EPOCHS):
        for i, m in enumerate(memory):
            # Pass 4 values, but we only need v_s and logits for loss
            v_s, logits, _, _ = model(m['op_t'], m['mc_t'], m['edge_t'], m['info'])
            
            dist = Categorical(torch.softmax(logits, dim=-1))
            log_prob = dist.log_prob(m['action'])
            entropy = dist.entropy()
            
            # Advantage
            advantage = returns[i] - v_s.detach()
            
            # PPO Clipped Objective
            ratio = torch.exp(log_prob - m['log_prob'].detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantage
            
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = 0.5 * F.mse_loss(v_s, returns[i].unsqueeze(0))
            
            # Total loss (Maximize entropy for exploration)
            loss = actor_loss + critic_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()






def evaluate_policy(env, model, n_episodes=5):
    episode_rewards = []
    episode_makespans = []
    product_completion = {
        'power_supply': [],
        'brake': [],
        'gear': []
    }
    
    model.eval()
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            op_t = torch.tensor(obs['op_nodes'], dtype=torch.float32)
            mc_t = torch.tensor(obs['mc_nodes'], dtype=torch.float32)
            raw_edges = obs['edge_index_om']
            mask = raw_edges[0] != -1
            edge_t = torch.tensor(raw_edges[:, mask], dtype=torch.long)
            
            with torch.no_grad():
                _, logits, _, _ = model(op_t, mc_t, edge_t, info)
                action = torch.argmax(logits).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # Collect metrics at end of episode
        episode_rewards.append(ep_reward)
        # current_time is our Makespan
        episode_makespans.append(env.unwrapped.current_time)
        
        inner_env = env.unwrapped
        for job in inner_env.jobs:
            finish_pct = (job['step'] / len(job['ops_indices'])) * 100
            product_completion[job['type']].append(finish_pct)

    model.train()
    
    stats = {
        'reward_mean': np.mean(episode_rewards),
        'reward_std': np.std(episode_rewards),
        'makespan_mean': np.mean(episode_makespans),
        'avg_finish_power_supply': np.mean(product_completion['power_supply']),
        'avg_finish_brake': np.mean(product_completion['brake']),
        'avg_finish_gear': np.mean(product_completion['gear'])
    }
    
    return stats




if __name__ == "__main__":
    train()





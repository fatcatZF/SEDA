import numpy as np 

import pandas as pd 

from collections import deque

import h2o

import gymnasium as gym 

from sklearn.metrics import roc_auc_score

from stable_baselines3 import PPO, SAC, DQN 

import joblib

import os 
import glob 

import json 

from environment_util import make_env 

import argparse

from river import drift 

import warnings
warnings.filterwarnings(action="ignore")

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole", help="name of environment")
    parser.add_argument("--policy-type", type=str, default="dqn", help="type of rl policy")
    parser.add_argument("--model-type", type=str, default="model", help="type of drift detector models")
    parser.add_argument("--env0-steps", type=int, default=1000, help="Validation Steps")
    parser.add_argument("--env1-steps", type=int, default=3000, help="Undrifted Steps")
    parser.add_argument("--env2-steps", type=int, default=3000, help="Semantic Drift Steps")
    parser.add_argument("--env3-steps", type=int, default=3000, help="Noisy Drift Steps")
    parser.add_argument("--n-exp-per-model", type=int, default=10, 
                        help="number of experiments of each trained model.") 

    args = parser.parse_args() 

    allowed_envs = {"cartpole", "mountaincar", "lunarlander", "hopper", 
                    "halfcheetah", "humanoid"}
    
    allowed_policy_types = {"dqn", "ppo", "sac"}

    if args.env not in allowed_envs:
        raise NotImplementedError(f"The environment {args.env} is not supported.")
    if args.policy_type not in allowed_policy_types:
        raise NotImplementedError(f"The policy {args.policy_type} is not supported.")

    print("Parsed arguments: ")
    print(args) 

    h2o.init(max_mem_size='4G')

    env_dict = {
      "cartpole" : "CartPole-v1",
      "mountaincar": "MountainCar-v0",
      "lunarlander": "LunarLander-v3",
      "hopper": "Hopper-v5",
      "halfcheetah": "HalfCheetah-v5",
      "humanoid": "Humanoid-v5"
    }


    # Load trained Agent
    if (args.policy_type=="dqn"):
        AGENT = DQN 
    elif (args.policy_type=="ppo"):
        AGENT = PPO 
    else:
        AGENT = SAC 

    policy_env_name = args.policy_type + '-' + args.env

    agent_path = os.path.join('./agents/', policy_env_name)
    agent = AGENT.load(agent_path) 
    print("Successfully Load Trained Agent.")

    env_action_discrete = {
        "cartpole": True,
        "mountaincar":True, 
        "lunarlander": True,
        "hopper": False,
        "halfcheetah": False,
        "humanoid": False
    }

    discrete_action = env_action_discrete[args.env]

    model_folder = os.path.join("models", args.env)

    if args.model_type == "model":
        pattern = "model_[0-9]"
    elif args.model_type == "model_od":
        pattern = "model_[0-9]_od"
    elif args.model_type == "model_on":
        pattern = "model_[0-9]_on"

    model_pattern = os.path.join(model_folder, pattern)
    matching_models = glob.glob(model_pattern)

    print(matching_models)
    if len(matching_models)==0:
        raise NotImplementedError(f"There is no trained model for the environment {args.env}.")


    single_best_pattern = "single_best_*"
    ens_best_pattern = "ens_best"
    ens_all_pattern = "ens_all"
    scaler_pattern = "scaler.pkl"
    pca_pattern = "pca.pkl"

    scalers = []
    pcas = []
    singles_best = []
    ensembles_best = []
    ensembles_all = []

    for folder in matching_models:
        #load scaler
        matches_scaler = glob.glob(os.path.join(folder, scaler_pattern))
        if not matches_scaler:
            raise FileNotFoundError("No scaler found")
        scaler_path = sorted(matches_scaler)[0]
        scalers.append(joblib.load(scaler_path))

        # load pca
        matches_pca = glob.glob(os.path.join(folder, pca_pattern))
        if not matches_pca:
            raise FileNotFoundError("No pca found")
        pca_path = sorted(matches_pca)[0]
        pcas.append(joblib.load(pca_path))

        # load single best model
        matches_single = glob.glob(os.path.join(folder, single_best_pattern))
        if not matches_single:
            raise FileNotFoundError("No single model found")
        single_best_path = sorted(matches_single)[0]
        singles_best.append(h2o.load_model(single_best_path))

        # load staked ensemble of best family
        matches_ens_best = glob.glob(os.path.join(folder, ens_best_pattern))
        if not matches_ens_best:
            raise FileNotFoundError("No ensemble best found")
        ens_best_path = sorted(matches_ens_best)[0]
        ensembles_best.append(h2o.load_model(ens_best_path))

        # load staked ensemble of all models
        matches_ens_all = glob.glob(os.path.join(folder, ens_all_pattern))
        if not matches_ens_all:
            raise FileNotFoundError("No ensemble all found")
        ens_all_path = sorted(matches_ens_all)[0]
        ensembles_all.append(h2o.load_model(ens_all_path))

    
    print("Number of loaded scalers: ", len(scalers))
    print("Number of loaded pcas: ", len(pcas))
    print("Number of loaded best single models: ", len(singles_best))
    print("Number of loaded staked ensembles from best: ", len(ensembles_best))
    print("Number of loaded staked ensembles of all models: ", len(ensembles_all))
       

    result = {}
    result_folder = f"./results/{args.env}"
    os.makedirs(result_folder, exist_ok=True) 
    result_path = os.path.join(result_folder, f"{args.model_type}.json")

    single_best_aucs_sem = []
    single_best_aucs_noise = []
    single_best_delays_ph_sem = []
    single_best_alarms_ph_sem = []
    single_best_delays_ph_noise = []
    single_best_alarms_ph_noise = []
    single_best_delays_ad_sem = []
    single_best_alarms_ad_sem = []
    single_best_delays_ad_noise = []
    single_best_alarms_ad_noise = []
    single_best_delays_ks_sem = []
    single_best_alarms_ks_sem = []
    single_best_delays_ks_noise = []
    single_best_alarms_ks_noise = []

    ens_best_aucs_sem = []
    ens_best_aucs_noise = []
    ens_best_delays_ph_sem = []
    ens_best_alarms_ph_sem = []
    ens_best_delays_ph_noise = []
    ens_best_alarms_ph_noise = []
    ens_best_delays_ad_sem = []
    ens_best_alarms_ad_sem = []
    ens_best_delays_ad_noise = []
    ens_best_alarms_ad_noise = []
    ens_best_delays_ks_sem = []
    ens_best_alarms_ks_sem = []
    ens_best_delays_ks_noise = []
    ens_best_alarms_ks_noise = []
    

    ens_all_aucs_sem = []
    ens_all_aucs_noise = []
    ens_all_delays_ph_sem = []
    ens_all_alarms_ph_sem = [] 
    ens_all_delays_ph_noise = []
    ens_all_alarms_ph_noise = [] 
    ens_all_delays_ad_sem = []
    ens_all_alarms_ad_sem = []
    ens_all_delays_ad_noise = []
    ens_all_alarms_ad_noise = []
    ens_all_delays_ks_sem = []
    ens_all_alarms_ks_sem = [] 
    ens_all_delays_ks_noise = []
    ens_all_alarms_ks_noise = [] 

    # create environment
    env0, env1, env2, env3 = make_env(name=args.env)
    print("Successfully create environments")

    for _ in range(args.n_exp_per_model):

        transitions_val = [] # For validation
        actions_val = []
        transitions_sem = [] # Semantic Drifts
        actions_sem = []
        transitions_noise = [] # Noisy Observations
        actions_noise = []
        
        # Validation steps
        env_current = env0 
        obs_t, _ = env_current.reset()
        for _ in range(args.env0_steps):
            action_t, _states = agent.predict(obs_t, deterministic=True)
            obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
            done = terminated or truncated
            transition = np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1)
            transition = transition.reshape(1, -1)
            transitions_val.append(transition)
            actions_val.append(action_t)

            obs_t = obs_tplus1
            if done:
                obs_t, _ = env_current.reset()
        
        transitions_val = np.concatenate(transitions_val, axis=0)
        actions_val = np.array(actions_val)

        # Semantic Drift
        env_current = env1 
        obs_t, _ = env_current.reset()
        total_steps = args.env1_steps + args.env2_steps
        for t in range(1, total_steps+1):
            action_t, _state = agent.predict(obs_t, deterministic=True)
            obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
            done = terminated or truncated
            transition = np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1)
            transition = transition.reshape(1, -1)
            transitions_sem.append(transition)
            actions_sem.append(action_t)

            obs_t = obs_tplus1
            if done:
                obs_t, _ = env_current.reset()
            if t == args.env1_steps:
                env_current = env2 
                obs_t, _ = env_current.reset()
        
        transitions_sem = np.concatenate(transitions_sem, axis=0)
        actions_sem = np.array(actions_sem)
        
        # Noisy Observations
        env_current = env1 
        obs_t, _ = env_current.reset()
        total_steps = args.env1_steps + args.env3_steps
        for t in range(1, total_steps+1):
            action_t, _state = agent.predict(obs_t, deterministic=True)
            obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
            done = terminated or truncated
            transition = np.concatenate([obs_t, obs_tplus1-obs_t], axis=-1)
            transition = transition.reshape(1, -1)
            transitions_noise.append(transition)
            actions_noise.append(action_t)

            obs_t = obs_tplus1
            if done:
                obs_t, _ = env_current.reset()
            if t == args.env1_steps:
                env_current = env3
                obs_t, _ = env_current.reset()

        transitions_noise = np.concatenate(transitions_noise, axis=0)
        actions_noise = np.array(actions_noise)

        
        y_env1 = np.zeros(args.env1_steps)
        y_env2 = np.ones(args.env2_steps)
        y_env3 = np.ones(args.env3_steps)

        for i in range(len(scalers)):
            scaler = scalers[i]
            pca = pcas[i]
            single_best = singles_best[i]
            ens_best = ensembles_best[i]
            ens_all = ensembles_all[i]

            if discrete_action:
               X_val = transitions_val
               
               X_val_scaled = scaler.transform(X_val)
               X_val_pca = pca.transform(X_val_scaled)
               df_val = pd.DataFrame(X_val_pca)
               df_val["action"] = actions_val
               df_val["action"] = df_val["action"].astype("category")

               X_sem = transitions_sem
               X_sem_scaled = scaler.transform(X_sem)
               X_sem_pca = pca.transform(X_sem_scaled)
               df_sem = pd.DataFrame(X_sem_pca)
               df_sem["action"] = actions_sem
               df_sem["action"] = df_sem["action"].astype("category")

               X_noise = transitions_noise
               X_noise_scaled = scaler.transform(X_noise)
               X_noise_pca = pca.transform(X_noise_scaled)
               df_noise = pd.DataFrame(X_noise_pca)
               df_noise["action"] = actions_noise
               df_noise["action"] = df_noise["action"].astype("category")

            else:
                X_val = np.concatenate([transitions_val, actions_val], axis=-1)
                X_val_scaled = scaler.transform(X_val)
                X_val_pca = pca.transform(X_val_scaled)
                df_val = pd.DataFrame(X_val_pca)

                X_sem = np.concatenate([transitions_sem, actions_sem], axis=-1)
                X_sem_scaled = scaler.transform(X_sem)
                X_sem_pca = pca.transform(X_sem_scaled)
                df_sem = pd.DataFrame(X_sem_pca)

                X_noise = np.concatenate([transitions_noise, actions_noise], axis=-1)
                X_noise_scaled = scaler.transform(X_noise)
                X_noise_pca = pca.transform(X_noise_scaled)
                df_noise = pd.DataFrame(X_noise_pca)
            

            hf_val = h2o.H2OFrame(df_val)
            hf_sem = h2o.H2OFrame(df_sem)
            hf_noise = h2o.H2OFrame(df_noise)

 
            # single best model
            preds_val = single_best.predict(hf_val)
            scores_drift_val = preds_val.as_data_frame(use_pandas=True)["True"].values
            mu_val = np.mean(scores_drift_val)
            sigma_val = np.std(scores_drift_val)

            preds_sem = single_best.predict(hf_sem)
            scores_drift_sem = preds_sem.as_data_frame(use_pandas=True)["True"].values
            scores_drift_sem = (scores_drift_sem - mu_val)/(sigma_val + 1e-8)

            ## compute sem auc
            y = np.concatenate([y_env1, y_env2])
            single_best_aucs_sem.append(roc_auc_score(y, scores_drift_sem))

            ## Page-Hinkley Semantic
            ph = drift.PageHinkley(mode='up', delta=0.005)
            fa = 0
            delay = args.env2_steps + 1000
            for t, val in enumerate(scores_drift_sem):
                ph.update(val)
                if ph.drift_detected and val > 0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 
            single_best_alarms_ph_sem.append(fa)
            single_best_delays_ph_sem.append(delay)


            ## ADWIN Semantic
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env2_steps + 1000
            for t, val in enumerate(scores_drift_sem):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
            single_best_alarms_ad_sem.append(fa)
            single_best_delays_ad_sem.append(delay)

            ## KSWIN semantic
            window = ((scores_drift_val-mu_val)/(sigma_val+1e-8)).tolist()
            kswin = drift.KSWIN(window=window)
            fa = 0
            delay = args.env2_steps+1000
            for t, val in enumerate(scores_drift_sem):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                            fa += 1
                    if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
            single_best_alarms_ks_sem.append(fa)
            single_best_delays_ks_sem.append(delay)


            
            preds_noise = single_best.predict(hf_noise)
            scores_drift_noise = preds_noise.as_data_frame(use_pandas=True)["True"].values
            scores_drift_noise = (scores_drift_noise - mu_val)/(sigma_val + 1e-8)

            ## compute noise auc
            y = np.concatenate([y_env1, y_env3])
            single_best_aucs_noise.append(roc_auc_score(y, scores_drift_noise))

            ## Page-Hinkley Noise
            ph = drift.PageHinkley(mode='up', delta=0.005)
            fa = 0
            delay = args.env3_steps + 1000
            for t, val in enumerate(scores_drift_noise):
                ph.update(val)
                if ph.drift_detected and val > 0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 
            single_best_alarms_ph_noise.append(fa)
            single_best_delays_ph_noise.append(delay)


            ## ADWIN Noise
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env3_steps + 1000
            for t, val in enumerate(scores_drift_noise):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
            single_best_alarms_ad_noise.append(fa)
            single_best_delays_ad_noise.append(delay)


            ## KSWIN semantic
            window = ((scores_drift_val-mu_val)/(sigma_val+1e-8)).tolist()
            kswin = drift.KSWIN(window=window)
            fa = 0
            delay = args.env3_steps+1000
            for t, val in enumerate(scores_drift_noise):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                            fa += 1
                    if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
            single_best_alarms_ks_noise.append(fa)
            single_best_delays_ks_noise.append(delay)




            # ens best model
            preds_val = ens_best.predict(hf_val)
            scores_drift_val = preds_val.as_data_frame(use_pandas=True)["True"].values
            mu_val = np.mean(scores_drift_val)
            sigma_val = np.std(scores_drift_val)

            preds_sem = ens_best.predict(hf_sem)
            scores_drift_sem = preds_sem.as_data_frame(use_pandas=True)["True"].values
            scores_drift_sem = (scores_drift_sem - mu_val)/(sigma_val + 1e-8)

            ## compute sem auc
            y = np.concatenate([y_env1, y_env2])
            ens_best_aucs_sem.append(roc_auc_score(y, scores_drift_sem))

            ## Page-Hinkley Semantic
            ph = drift.PageHinkley(mode='up', delta=0.005)
            fa = 0
            delay = args.env2_steps + 1000
            for t, val in enumerate(scores_drift_sem):
                ph.update(val)
                if ph.drift_detected and val > 0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 
            ens_best_alarms_ph_sem.append(fa)
            ens_best_delays_ph_sem.append(delay)


            ## ADWIN Semantic
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env2_steps + 1000
            for t, val in enumerate(scores_drift_sem):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
            ens_best_alarms_ad_sem.append(fa)
            ens_best_delays_ad_sem.append(delay)

            ## KSWIN semantic
            window = ((scores_drift_val-mu_val)/(sigma_val+1e-8)).tolist()
            kswin = drift.KSWIN(window=window)
            fa = 0
            delay = args.env2_steps+1000
            for t, val in enumerate(scores_drift_sem):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                            fa += 1
                    if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
            ens_best_alarms_ks_sem.append(fa)
            ens_best_delays_ks_sem.append(delay)


            
            preds_noise = ens_best.predict(hf_noise)
            scores_drift_noise = preds_noise.as_data_frame(use_pandas=True)["True"].values
            scores_drift_noise = (scores_drift_noise - mu_val)/(sigma_val + 1e-8)

            ## compute noise auc
            y = np.concatenate([y_env1, y_env3])
            ens_best_aucs_noise.append(roc_auc_score(y, scores_drift_noise))

            ## Page-Hinkley Noise
            ph = drift.PageHinkley(mode='up', delta=0.005)
            fa = 0
            delay = args.env3_steps + 1000
            for t, val in enumerate(scores_drift_noise):
                ph.update(val)
                if ph.drift_detected and val > 0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 
            ens_best_alarms_ph_noise.append(fa)
            ens_best_delays_ph_noise.append(delay)


            ## ADWIN Noise
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env3_steps + 1000
            for t, val in enumerate(scores_drift_noise):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
            ens_best_alarms_ad_noise.append(fa)
            ens_best_delays_ad_noise.append(delay)


            ## KSWIN semantic
            window = ((scores_drift_val-mu_val)/(sigma_val+1e-8)).tolist()
            kswin = drift.KSWIN(window=window)
            fa = 0
            delay = args.env3_steps+1000
            for t, val in enumerate(scores_drift_noise):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                            fa += 1
                    if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
            ens_best_alarms_ks_noise.append(fa)
            ens_best_delays_ks_noise.append(delay)






            # ens all model
            preds_val = ens_all.predict(hf_val)
            scores_drift_val = preds_val.as_data_frame(use_pandas=True)["True"].values
            mu_val = np.mean(scores_drift_val)
            sigma_val = np.std(scores_drift_val)

            preds_sem = ens_all.predict(hf_sem)
            scores_drift_sem = preds_sem.as_data_frame(use_pandas=True)["True"].values
            scores_drift_sem = (scores_drift_sem - mu_val)/(sigma_val + 1e-8)

            ## compute sem auc
            y = np.concatenate([y_env1, y_env2])
            ens_all_aucs_sem.append(roc_auc_score(y, scores_drift_sem))

            ## Page-Hinkley Semantic
            ph = drift.PageHinkley(mode='up', delta=0.005)
            fa = 0
            delay = args.env2_steps + 1000
            for t, val in enumerate(scores_drift_sem):
                ph.update(val)
                if ph.drift_detected and val > 0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 
            ens_all_alarms_ph_sem.append(fa)
            ens_all_delays_ph_sem.append(delay)


            ## ADWIN Semantic
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env2_steps + 1000
            for t, val in enumerate(scores_drift_sem):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
            ens_all_alarms_ad_sem.append(fa)
            ens_all_delays_ad_sem.append(delay)

            ## KSWIN semantic
            window = ((scores_drift_val-mu_val)/(sigma_val+1e-8)).tolist()
            kswin = drift.KSWIN(window=window)
            fa = 0
            delay = args.env2_steps+1000
            for t, val in enumerate(scores_drift_sem):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                            fa += 1
                    if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
            ens_all_alarms_ks_sem.append(fa)
            ens_all_delays_ks_sem.append(delay)


            
            preds_noise = ens_all.predict(hf_noise)
            scores_drift_noise = preds_noise.as_data_frame(use_pandas=True)["True"].values
            scores_drift_noise = (scores_drift_noise - mu_val)/(sigma_val + 1e-8)

            ## compute noise auc
            y = np.concatenate([y_env1, y_env3])
            ens_all_aucs_noise.append(roc_auc_score(y, scores_drift_noise))

            ## Page-Hinkley Noise
            ph = drift.PageHinkley(mode='up', delta=0.005)
            fa = 0
            delay = args.env3_steps + 1000
            for t, val in enumerate(scores_drift_noise):
                ph.update(val)
                if ph.drift_detected and val > 0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 
            ens_all_alarms_ph_noise.append(fa)
            ens_all_delays_ph_noise.append(delay)


            ## ADWIN Noise
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env3_steps + 1000
            for t, val in enumerate(scores_drift_noise):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                        if t<args.env1_steps:
                           fa+=1
                        if t>=args.env1_steps:
                           delay = t-args.env1_steps
                           break
                
            ens_all_alarms_ad_noise.append(fa)
            ens_all_delays_ad_noise.append(delay)


            ## KSWIN noise
            window = ((scores_drift_val-mu_val)/(sigma_val+1e-8)).tolist()
            kswin = drift.KSWIN(window=window)
            fa = 0
            delay = args.env3_steps+1000
            for t, val in enumerate(scores_drift_noise):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                            fa += 1
                    if t >= args.env1_steps:
                            delay = t - args.env1_steps
                
            ens_all_alarms_ks_noise.append(fa)
            ens_all_delays_ks_noise.append(delay)

    
    result["single_best"] = {"sem_auc": np.mean(single_best_aucs_sem),
                             "noise_auc": np.mean(single_best_aucs_noise),
                             "sem_fa_ph": np.mean(single_best_alarms_ph_sem),
                             "sem_delay_ph": np.mean(single_best_delays_ph_sem),
                             "sem_fa_ad": np.mean(single_best_alarms_ad_sem),
                             "sem_delay_ad": np.mean(single_best_delays_ad_sem),
                             "sem_fa_ks": np.mean(single_best_alarms_ks_sem),
                             "sem_delay_ks": np.mean(single_best_delays_ks_sem),
                             "noise_fa_ph": np.mean(single_best_alarms_ph_noise),
                             "noise_delay_ph": np.mean(single_best_delays_ph_noise),
                             "noise_fa_ad": np.mean(single_best_alarms_ad_noise),
                             "noise_delay_ad": np.mean(single_best_delays_ad_noise),
                             "noise_fa_ks": np.mean(single_best_alarms_ks_noise),
                             "noise_delay_ks": np.mean(single_best_delays_ks_noise)}
    
    result["ens_best"] = {"sem_auc": np.mean(ens_best_aucs_sem),
                          "noise_auc": np.mean(ens_best_aucs_noise),
                             "sem_fa_ph": np.mean(ens_best_alarms_ph_sem),
                             "sem_delay_ph": np.mean(ens_best_delays_ph_sem),
                             "sem_fa_ad": np.mean(ens_best_alarms_ad_sem),
                             "sem_delay_ad": np.mean(ens_best_delays_ad_sem),
                             "sem_fa_ks": np.mean(ens_best_alarms_ks_sem),
                             "sem_delay_ks": np.mean(ens_best_delays_ks_sem),
                             "noise_fa_ph": np.mean(ens_best_alarms_ph_noise),
                             "noise_delay_ph": np.mean(ens_best_delays_ph_noise),
                             "noise_fa_ad": np.mean(ens_best_alarms_ad_noise),
                             "noise_delay_ad": np.mean(ens_best_delays_ad_noise),
                             "noise_fa_ks": np.mean(ens_best_alarms_ks_noise),
                             "noise_delay_ks": np.mean(ens_best_delays_ks_noise)}
    

    result["ens_all"] = {"sem_auc": np.mean(ens_all_aucs_sem),
                         "noise_auc": np.mean(ens_all_aucs_noise),
                             "sem_fa_ph": np.mean(ens_all_alarms_ph_sem),
                             "sem_delay_ph": np.mean(ens_all_delays_ph_sem),
                             "sem_fa_ad": np.mean(ens_all_alarms_ad_sem),
                             "sem_delay_ad": np.mean(ens_all_delays_ad_sem),
                             "sem_fa_ks": np.mean(ens_all_alarms_ks_sem),
                             "sem_delay_ks": np.mean(ens_all_delays_ks_sem),
                             "noise_fa_ph": np.mean(ens_all_alarms_ph_noise),
                             "noise_delay_ph": np.mean(ens_all_delays_ph_noise),
                             "noise_fa_ad": np.mean(ens_all_alarms_ad_noise),
                             "noise_delay_ad": np.mean(ens_all_delays_ad_noise),
                             "noise_fa_ks": np.mean(ens_all_alarms_ks_noise),
                             "noise_delay_ks": np.mean(ens_all_delays_ks_noise)}
    

    print("number of single best semantic aucs: ", len(single_best_aucs_sem))
    print("number of single best semantic ph fas: ", len(single_best_alarms_ph_sem))
    print("number of single best semantic ph delays: ", len(single_best_delays_ph_sem))
    print("number of single best semantic ad fas: ", len(single_best_alarms_ad_sem))
    print("number of single best semantic ad delays: ", len(single_best_delays_ad_sem))
    print("number of single best semantic ks fas: ", len(single_best_alarms_ks_sem))
    print("number of single best semantic ks delays: ", len(single_best_delays_ks_sem))

    print("number of single best noise aucs: ", len(single_best_aucs_noise))
    print("number of single best noise ph fas: ", len(single_best_alarms_ph_noise))
    print("number of single best noise ph delays: ", len(single_best_delays_ph_noise))
    print("number of single best noise ad fas: ", len(single_best_alarms_ad_noise))
    print("number of single best noise ad delays: ", len(single_best_delays_ad_noise))
    print("number of single best noise ks fas: ", len(single_best_alarms_ks_noise))
    print("number of single best noise ks delays: ", len(single_best_delays_ks_noise))

    print("number of ens best semantic aucs: ", len(ens_best_aucs_sem))
    print("number of ens best semantic ph fas: ", len(ens_best_alarms_ph_sem))
    print("number of ens best semantic ph delays: ", len(ens_best_delays_ph_sem))
    print("number of ens best semantic ad fas: ", len(ens_best_alarms_ad_sem))
    print("number of ens best semantic ad delays: ", len(ens_best_delays_ad_sem))
    print("number of ens best semantic ks fas: ", len(ens_best_alarms_ks_sem))
    print("number of ens best semantic ks delays: ", len(ens_best_delays_ks_sem))

    print("number of ens best noise aucs: ", len(ens_best_aucs_noise))
    print("number of ens best noise ph fas: ", len(ens_best_alarms_ph_noise))
    print("number of ens best noise ph delays: ", len(ens_best_delays_ph_noise))
    print("number of ens best noise ad fas: ", len(ens_best_alarms_ad_noise))
    print("number of ens best noise ad delays: ", len(ens_best_delays_ad_noise))
    print("number of ens best noise ks fas: ", len(ens_best_alarms_ks_noise))
    print("number of ens best noise ks delays: ", len(ens_best_delays_ks_noise))

    print("number of ens all semantic aucs: ", len(ens_all_aucs_sem))
    print("number of ens all semantic ph fas: ", len(ens_all_alarms_ph_sem))
    print("number of ens all semantic ph delays: ", len(ens_all_delays_ph_sem))
    print("number of ens all semantic ad fas: ", len(ens_all_alarms_ad_sem))
    print("number of ens all semantic ad delays: ", len(ens_all_delays_ad_sem))
    print("number of ens all semantic ks fas: ", len(ens_all_alarms_ks_sem))
    print("number of ens all semantic ks delays: ", len(ens_all_delays_ks_sem))

    print("number of ens all noise aucs: ", len(ens_all_aucs_noise))
    print("number of ens all noise ph fas: ", len(ens_all_alarms_ph_noise))
    print("number of ens all noise ph delays: ", len(ens_all_delays_ph_noise))
    print("number of ens all noise ad fas: ", len(ens_all_alarms_ad_noise))
    print("number of ens all noise ad delays: ", len(ens_all_delays_ad_noise))
    print("number of ens all noise ks fas: ", len(ens_all_alarms_ks_noise))
    print("number of ens all noise ks delays: ", len(ens_all_delays_ks_noise))

    
    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))



    h2o.cluster().shutdown(prompt=False)




    


if __name__ == "__main__":
    main()
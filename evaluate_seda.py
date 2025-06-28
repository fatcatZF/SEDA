import numpy as np 
from collections import deque

import h2o

import gymnasium as gym 

from sklearn.metrics import roc_auc_score

from stable_baselines3 import PPO, SAC, DQN 

import torch 
import joblib

import os 
import glob 
import pickle
import json 

from environment_util import make_env 

import argparse

from river import drift 

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

    single_best_aucs = []
    single_best_delays_ph = []
    single_best_alarms_ph = []
    single_best_delays_ad = []
    single_best_alarms_ad = []
    single_best_delays_ks = []
    single_best_alarms_ks = []

    ens_best_aucs = []
    ens_best_delays_ph = []
    ens_best_alarms_ph = []
    ens_best_delays_ad = []
    ens_best_alarms_ad = []
    ens_best_delays_ks = []
    ens_best_alarms_ks = []
    

    ens_all_aucs = []
    ens_all_delays_ph = []
    ens_all_alarms_ph = [] 
    ens_all_delays_ad = []
    ens_all_alarms_ad = []
    ens_all_delays_ks = []
    ens_all_alarms_ks = [] 


    h2o.cluster().shutdown(prompt=False)




    


if __name__ == "__main__":
    main()
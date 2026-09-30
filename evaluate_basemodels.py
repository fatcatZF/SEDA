import os
import glob
import json
import argparse
import warnings

import numpy as np
import pandas as pd
import h2o
import joblib
from sklearn.metrics import roc_auc_score

from stable_baselines3 import PPO, SAC, DQN
from environment_util import make_env

warnings.filterwarnings(action="ignore")


def rollout(env, agent, n_steps, obs_t):
    """Placeholder-free helper is avoided on purpose: rollouts are inlined in main
    to mirror evaluate_seda.py exactly."""
    raise NotImplementedError


def collect_transitions(agent, env_first, env_second, n_first, n_second):
    """Run `n_first` steps in env_first, then `n_second` steps in env_second.
    Returns (transitions, actions). Mirrors the loops in evaluate_seda.py."""
    transitions, actions = [], []
    env_current = env_first
    obs_t, _ = env_current.reset()
    for t in range(1, n_first + n_second + 1):
        action_t, _ = agent.predict(obs_t, deterministic=True)
        obs_tp1, _, terminated, truncated, _ = env_current.step(action_t)
        done = terminated or truncated
        transition = np.concatenate([obs_t, obs_tp1 - obs_t], axis=-1).reshape(1, -1)
        transitions.append(transition)
        actions.append(action_t)

        obs_t = obs_tp1
        if done:
            obs_t, _ = env_current.reset()
        if t == n_first:
            env_current = env_second
            obs_t, _ = env_current.reset()
    return np.concatenate(transitions, axis=0), np.array(actions)


def build_frame(transitions, actions, scaler, pca, discrete_action):
    """Scale -> PCA -> H2OFrame, same as evaluate_seda.py."""
    if discrete_action:
        X = transitions
    else:
        X = np.concatenate([transitions, actions], axis=-1)
    df = pd.DataFrame(pca.transform(scaler.transform(X)))
    if discrete_action:
        df["action"] = actions
        df["action"] = df["action"].astype("category")
    return h2o.H2OFrame(df)


def score_true(model, hf):
    preds = model.predict(hf)
    return preds.as_data_frame(use_pandas=True)["True"].values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole", help="name of environment")
    parser.add_argument("--policy-type", type=str, default="dqn", help="type of rl policy")
    parser.add_argument("--model-type", type=str, default="model", help="type of drift detector models")
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

    h2o.init(max_mem_size='20G')

    # Load trained agent
    AGENT = {"dqn": DQN, "ppo": PPO, "sac": SAC}[args.policy_type]
    agent_path = os.path.join('./agents/', args.policy_type + '-' + args.env)
    agent = AGENT.load(agent_path)
    print("Successfully Load Trained Agent.")

    env_action_discrete = {
        "cartpole": True, "mountaincar": True, "lunarlander": True,
        "hopper": False, "halfcheetah": False, "humanoid": False,
    }
    discrete_action = env_action_discrete[args.env]

    # Locate trained detector folders
    model_folder = os.path.join("models", args.env)
    patterns = {
        "model": "model_[0-9]",
        "model_od": "model_[0-9]_od",
        "model_on": "model_[0-9]_on",
    }
    if args.model_type not in patterns:
        raise NotImplementedError(f"Unknown model type {args.model_type}")
    matching_models = sorted(glob.glob(os.path.join(model_folder, patterns[args.model_type])))
    print(matching_models)
    if len(matching_models) == 0:
        raise NotImplementedError(f"There is no trained model for the environment {args.env}.")

    # Load scaler, pca and ens_all of every folder, then collect the base models
    scalers, pcas, base_models_per_folder, folder_names = [], [], [], []
    for folder in matching_models:
        scaler_path = os.path.join(folder, "scaler.pkl")
        pca_path = os.path.join(folder, "pca.pkl")
        ens_all_path = os.path.join(folder, "ens_all")
        for p in (scaler_path, pca_path, ens_all_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found")

        scalers.append(joblib.load(scaler_path))
        pcas.append(joblib.load(pca_path))
        ens_all = h2o.load_model(ens_all_path)

        base_models = []
        for mid in ens_all.base_models:
            m = h2o.get_model(mid)  # base models are restored together with the ensemble
            base_models.append({
                "id": mid,
                "algo": m.algo,                         # gbm, drf, xgboost, glm, deeplearning
                "family": mid.split("_")[0],            # keeps XRT separate from DRF
                "time_s": (m._model_json["output"].get("run_time") or 0) / 1000,
                "model": m,
            })
        base_models_per_folder.append(base_models)
        folder_names.append(os.path.basename(folder))
        print(f"{folder}: loaded {len(base_models)} base models")

    # Environments
    env0, env1, env2, env3 = make_env(name=args.env)  # env0 (validation) is not needed here
    print("Successfully create environments")

    # Labels: 0 for undrifted steps, 1 for drifted steps
    y_sem = np.concatenate([np.zeros(args.env1_steps), np.ones(args.env2_steps)])
    y_noise = np.concatenate([np.zeros(args.env1_steps), np.ones(args.env3_steps)])

    rows = []
    for exp in range(args.n_exp_per_model):
        # Same rollouts are shared by all folders and all base models within an experiment
        transitions_sem, actions_sem = collect_transitions(
            agent, env1, env2, args.env1_steps, args.env2_steps)
        transitions_noise, actions_noise = collect_transitions(
            agent, env1, env3, args.env1_steps, args.env3_steps)

        for i, folder_name in enumerate(folder_names):
            hf_sem = build_frame(transitions_sem, actions_sem, scalers[i], pcas[i], discrete_action)
            hf_noise = build_frame(transitions_noise, actions_noise, scalers[i], pcas[i], discrete_action)

            for bm in base_models_per_folder[i]:
                # AUC is rank-based, so no need to standardise scores with validation stats
                scores_sem = score_true(bm["model"], hf_sem)
                scores_noise = score_true(bm["model"], hf_noise)
                rows.append({
                    "env": args.env,
                    "model_type": args.model_type,
                    "folder": folder_name,
                    "exp": exp,
                    "model_id": bm["id"],
                    "algo": bm["algo"],
                    "family": bm["family"],
                    "auc_sem": roc_auc_score(y_sem, scores_sem),
                    "auc_noise": roc_auc_score(y_noise, scores_noise),
                    "time_s": bm["time_s"],
                })

            h2o.remove(hf_sem)
            h2o.remove(hf_noise)
        print(f"Experiment {exp + 1}/{args.n_exp_per_model} done")

    df = pd.DataFrame(rows)

    result_folder = f"./results/{args.env}"
    os.makedirs(result_folder, exist_ok=True)

    # Per-base-model results (raw, one row per model x folder x experiment)
    csv_path = os.path.join(result_folder, f"basemodels_{args.model_type}.csv")
    df.to_csv(csv_path, index=False)

    # Family summary: average over experiments first, then over trained detector folders
    per_model = (df.groupby(["folder", "model_id", "algo"])
                   .agg(auc_sem=("auc_sem", "mean"),
                        auc_noise=("auc_noise", "mean"),
                        time_s=("time_s", "first"))
                   .reset_index())

    per_folder_family = (per_model.groupby(["folder", "algo"])
                         .agg(n_models=("model_id", "count"),
                              mean_auc_sem=("auc_sem", "mean"),
                              best_auc_sem=("auc_sem", "max"),
                              mean_auc_noise=("auc_noise", "mean"),
                              best_auc_noise=("auc_noise", "max"),
                              total_time_s=("time_s", "sum"))
                         .reset_index())

    summary = {}
    for algo, g in per_folder_family.groupby("algo"):
        summary[algo] = {
            col: {"mean": float(g[col].mean()), "std": float(g[col].std(ddof=1)) if len(g) > 1 else 0.0}
            for col in ["n_models", "mean_auc_sem", "best_auc_sem",
                        "mean_auc_noise", "best_auc_noise", "total_time_s"]
        }

    json_path = os.path.join(result_folder, f"basemodels_{args.model_type}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, separators=(',', ':'))

    print(per_folder_family.groupby("algo")[
        ["mean_auc_sem", "best_auc_sem", "mean_auc_noise", "best_auc_noise", "total_time_s"]
    ].mean().round(3))
    print(f"Saved {csv_path} and {json_path}")

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
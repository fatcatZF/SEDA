import os
import glob
import json
import argparse
import warnings

import pandas as pd
import h2o

warnings.filterwarnings(action="ignore")

PATTERNS = {
    "model": "model_[0-9]",
    "model_od": "model_[0-9]_od",
    "model_on": "model_[0-9]_on",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+",
                        default=["cartpole", "halfcheetah", "hopper", "humanoid", "lunarlander"],
                        help="environments to include")
    parser.add_argument("--model-type", type=str, default="model",
                        help="type of drift detector models (model, model_od, model_on)")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--out-dir", type=str, default="./results/basemodels_training_time")
    parser.add_argument("--include-metalearner", action="store_true",
                        help="also record the ensemble's own run_time (metalearner only)")
    args = parser.parse_args()

    if args.model_type not in PATTERNS:
        raise NotImplementedError(f"Unknown model type {args.model_type}")

    print("Parsed arguments: ")
    print(args)

    h2o.init(max_mem_size="20G")

    rows = []
    for env in args.envs:
        folders = sorted(glob.glob(os.path.join(args.models_dir, env, PATTERNS[args.model_type])))
        if not folders:
            print(f"[warning] no trained models for {env}, skipping")
            continue

        for folder in folders:
            ens_path = os.path.join(folder, "ens_all")
            if not os.path.exists(ens_path):
                print(f"[warning] {ens_path} not found, skipping")
                continue

            ens = h2o.load_model(ens_path)  # base models are restored with the ensemble
            folder_name = os.path.basename(folder)
            n_before = len(rows)

            for mid in ens.base_models:
                m = h2o.get_model(mid)
                run_time = m._model_json["output"].get("run_time")
                rows.append({
                    "env": env,
                    "folder": folder_name,
                    "model_id": mid,
                    "algo": m.algo,               # gbm, drf, xgboost, glm, deeplearning
                    "family": mid.split("_")[0],  # keeps XRT separate from DRF
                    "time_s": run_time / 1000 if run_time is not None else None,
                })

            if args.include_metalearner:
                run_time = ens._model_json["output"].get("run_time")
                rows.append({
                    "env": env,
                    "folder": folder_name,
                    "model_id": ens.model_id,
                    "algo": "stackedensemble",
                    "family": "StackedEnsemble",
                    "time_s": run_time / 1000 if run_time is not None else None,
                })

            print(f"{folder}: {len(rows) - n_before} models")
            h2o.remove_all()  # free memory before loading the next ensemble

    if not rows:
        raise RuntimeError("No models found.")

    df = pd.DataFrame(rows)
    missing = df["time_s"].isna().sum()
    if missing:
        print(f"[warning] {missing} models have no run_time and are ignored in the sums")

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, f"basemodels_time_raw_{args.model_type}.csv"), index=False)

    # Level 1: one row per (env, detector folder, family)
    per_folder = (df.groupby(["env", "folder", "algo"])
                    .agg(n_models=("model_id", "count"),
                         total_time_s=("time_s", "sum"),
                         mean_time_s=("time_s", "mean"),
                         max_time_s=("time_s", "max"))
                    .reset_index())

    # Level 2: average over the detector folders inside each environment
    per_env = (per_folder.groupby(["env", "algo"])
                         .agg(n_models=("n_models", "mean"),
                              total_time_s=("total_time_s", "mean"),
                              total_time_s_std=("total_time_s", "std"),
                              mean_time_s=("mean_time_s", "mean"),
                              mean_time_s_std=("mean_time_s", "std"),
                              max_time_s=("max_time_s", "mean"))
                         .reset_index())

    # Level 3: average over environments (each environment counts once)
    overall = (per_env.groupby("algo")
                      .agg(n_models=("n_models", "mean"),
                           total_time_s=("total_time_s", "mean"),
                           total_time_s_std=("total_time_s", "std"),
                           mean_time_s=("mean_time_s", "mean"),
                           mean_time_s_std=("mean_time_s", "std"),
                           max_time_s=("max_time_s", "mean"))
                      .reset_index())
    overall["share_of_total"] = overall["total_time_s"] / overall["total_time_s"].sum()
    overall = overall.sort_values("total_time_s", ascending=False)

    per_folder.to_csv(os.path.join(args.out_dir, f"basemodels_time_per_folder_{args.model_type}.csv"), index=False)
    per_env.to_csv(os.path.join(args.out_dir, f"basemodels_time_per_env_{args.model_type}.csv"), index=False)
    overall.to_csv(os.path.join(args.out_dir, f"basemodels_time_overall_{args.model_type}.csv"), index=False)

    with open(os.path.join(args.out_dir, f"basemodels_time_overall_{args.model_type}.json"), "w") as f:
        json.dump(overall.set_index("algo").fillna(0.0).to_dict(orient="index"), f, indent=2)

    pd.set_option("display.width", 200)
    print("\nPer environment (mean over detector folders), seconds:")
    print(per_env.pivot(index="algo", columns="env", values="total_time_s").round(1))
    print("\nOverall (mean over environments):")
    print(overall.round(3).to_string(index=False))
    print(f"\nSaved results to {args.out_dir}")

    h2o.cluster().shutdown(prompt=False)


if __name__ == "__main__":
    main()
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts

# ---------- settings ----------
envs   = ["cartpole", "lunarlander", "hopper", "halfcheetah", "humanoid"]
labels = ["CartPole", "LunarLander", "Hopper", "HalfCheetah", "Humanoid"]
names  = {"deeplearning": "Deep learning", "drf": "DRF", "gbm": "GBM",
          "glm": "GLM", "xgboost": "XGBoost"}

# ---------- load data ----------
raw_path = "results/basemodels_training_time/basemodels_time_raw_model.csv"
if os.path.exists(raw_path):
    df = pd.read_csv(raw_path)
else:
    # fallback: time_s is also stored in the evaluation CSVs (repeated for every experiment)
    files = glob.glob("results/*/basemodels_model.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["env", "folder", "model_id"])

df = df[df["env"].isin(envs)]
df = df[df["algo"] != "stackedensemble"].dropna(subset=["time_s"])

missing = set(envs) - set(df["env"].unique())
if missing:
    print(f"[warning] no results for: {sorted(missing)}")

# ---------- aggregate ----------
# 1) per (env, detector folder, family): total and mean time of the family
per_folder = (df.groupby(["env", "folder", "algo"])["time_s"]
                .agg(total_time_s="sum", mean_time_s="mean")
                .reset_index())

# 2) mean and std across detector folders
agg = (per_folder.groupby(["env", "algo"])[["total_time_s", "mean_time_s"]]
                 .agg(["mean", "std"]))

fams = sorted(per_folder["algo"].unique())

# ---------- plotting ----------
def plot_time(col, ylabel, filename):
    x = np.arange(len(envs))
    w = 0.8 / len(fams)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, f in enumerate(fams):
        d = agg.xs(f, level="algo").reindex(envs)
        mean = d[(col, "mean")].values
        std = d[(col, "std")].fillna(0).values
        # asymmetric error bars: keep the lower end positive for the log axis
        yerr = np.vstack([np.minimum(std, mean * 0.99), std])
        ax.bar(x + i * w, mean, w, yerr=yerr, capsize=2, label=names.get(f, f))

    ax.set_yscale("log")   # deep learning would flatten the other families
    ax.set_xticks(x + w * (len(fams) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=len(fams), loc="lower center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

plot_time("total_time_s", "Total training time per family (s, log)",
          "basemodel_time_total.pdf")
plot_time("mean_time_s", "Mean training time per model (s, log)",
          "basemodel_time_mean.pdf")

# ---------- exact values for a table ----------
table = agg.xs("mean", axis=1, level=1).unstack("algo")
table.to_csv("basemodel_time_table.csv")
print(table.round(1).to_string())
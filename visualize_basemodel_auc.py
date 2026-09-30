import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts (some publishers require it)

# ---------- settings ----------
envs   = ["cartpole", "lunarlander", "hopper", "halfcheetah", "humanoid"]
labels = ["CartPole", "LunarLander", "Hopper", "HalfCheetah", "Humanoid"]
names  = {"deeplearning": "Deep learning", "drf": "DRF", "gbm": "GBM",
          "glm": "GLM", "xgboost": "XGBoost"}

# ---------- load results ----------
files = glob.glob("results/*/basemodels_model.csv")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df = df[df["env"].isin(envs)]

missing = set(envs) - set(df["env"].unique())
if missing:
    print(f"[warning] no results for: {sorted(missing)}")

# ---------- aggregate ----------
# 1) average over experiments -> one value per (env, folder, model)
per_model = (df.groupby(["env", "folder", "model_id", "algo"])[["auc_sem", "auc_noise"]]
               .mean().reset_index())

# 2) mean AUC of each family inside one detector folder
per_folder = (per_model.groupby(["env", "folder", "algo"])[["auc_sem", "auc_noise"]]
                       .mean().reset_index())

# 3) mean and std across detector folders
agg = (per_folder.groupby(["env", "algo"])[["auc_sem", "auc_noise"]]
                 .agg(["mean", "std"]))

fams = sorted(per_folder["algo"].unique())

# ---------- plotting ----------
def plot_auc(col, filename):
    x = np.arange(len(envs))
    w = 0.8 / len(fams)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, f in enumerate(fams):
        d = agg.xs(f, level="algo").reindex(envs)
        ax.bar(x + i * w, d[(col, "mean")], w, yerr=d[(col, "std")],
               capsize=2, label=names.get(f, f))

    ax.axhline(0.5, color="gray", ls="--", lw=1)      # chance level
    ax.set_xticks(x + w * (len(fams) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean AUC of base models")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=len(fams), loc="lower center", bbox_to_anchor=(0.5, 1.02),
              frameon=False, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")

plot_auc("auc_sem",   "basemodel_auc_semantic.pdf")
plot_auc("auc_noise", "basemodel_auc_noisy.pdf")

# ---------- optional: exact values for a table ----------
table = agg.xs("mean", axis=1, level=1).unstack("algo")
table.to_csv("basemodel_auc_table.csv")
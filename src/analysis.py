import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import f_classif

from data import ROOT, get_feature_cols, load_data, split_by_session
from evaluate import compute_eer
from models import score_model

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

SIZES = [10, 25, 50, 100, 200]
SIZE_MODELS = ["ScaledManhattan", "Mahalanobis", "RandomForest"]


def feature_analysis(df, features):
    """Which timing features differ most between people (ANOVA F-score)?"""
    f_scores, _ = f_classif(df[features].values, df["subject"].values)
    scores = pd.Series(f_scores, index=features).sort_values(ascending=False)
    scores.to_csv(RESULTS / "feature_f_scores.csv", header=["f_score"])

    print("\n=== Top 10 most discriminative features ===")
    print(scores.head(10).round(1))

    groups = scores.groupby(lambda c: c.split(".")[0]).mean().round(1)
    print("\nMean F-score by feature type (H=hold, DD=down-down, UD=up-down):")
    print(groups)

    plt.figure(figsize=(8, 6))
    scores.head(15).sort_values().plot(kind="barh")
    plt.title("Top 15 features that distinguish people (ANOVA F-score)")
    plt.xlabel("F-score")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=150)
    plt.close()


def training_size_curve(df, features):
    """How does EER change with the number of genuine training samples?"""
    train, test = split_by_session(df)
    subjects = sorted(df["subject"].unique())
    rng = np.random.default_rng(42)
    rows = []

    for i, user in enumerate(subjects, 1):
        # Same impostor protocol as run_experiments.py
        others = [s for s in subjects if s != user]
        rng.shuffle(others)
        half = len(others) // 2
        imp_train_subj, imp_test_subj = others[:half], others[half:]

        X_gen_train = train.loc[train["subject"] == user, features].values
        X_gen_test = test.loc[test["subject"] == user, features].values
        X_imp_train = (
            train[train["subject"].isin(imp_train_subj)]
            .groupby("subject")
            .sample(n=8, random_state=0)[features]
            .values
        )
        X_imp_test = (
            test[test["subject"].isin(imp_test_subj)]
            .groupby("subject")
            .sample(n=20, random_state=0)[features]
            .values
        )
        X_test = np.vstack([X_gen_test, X_imp_test])
        y_test = np.r_[np.ones(len(X_gen_test)), np.zeros(len(X_imp_test))]

        for n in SIZES:
            idx = rng.choice(len(X_gen_train), size=n, replace=False)
            X_gen = X_gen_train[idx]
            for name in SIZE_MODELS:
                scores = score_model(name, X_gen, X_imp_train, X_test)
                rows.append(
                    {"n_train": n, "model": name, "eer": compute_eer(y_test, scores)}
                )
        print(f"[{i}/{len(subjects)}] training-size curve: {user}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "training_size_results.csv", index=False)
    table = res.groupby(["model", "n_train"])["eer"].mean().unstack().round(4)
    print("\n=== Mean EER vs number of genuine training samples ===")
    print(table)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=res, x="n_train", y="eer", hue="model", marker="o", errorbar=None)
    plt.title("EER vs number of genuine training samples")
    plt.xlabel("Genuine training samples")
    plt.ylabel("Mean EER (lower is better)")
    plt.tight_layout()
    plt.savefig(RESULTS / "eer_vs_training_size.png", dpi=150)
    plt.close()


def user_difficulty():
    """Which users are easiest and hardest to verify with the best model?"""
    per_user = pd.read_csv(RESULTS / "per_user_results.csv")
    comparison = pd.read_csv(RESULTS / "model_comparison.csv", index_col=0)
    best = comparison.index[0]
    u = per_user[per_user["model"] == best].sort_values("eer")

    print(f"\n=== User difficulty with best model: {best} ===")
    print("Easiest 5:\n", u.head(5)[["user", "eer"]].round(4).to_string(index=False))
    print("Hardest 5:\n", u.tail(5)[["user", "eer"]].round(4).to_string(index=False))

    plt.figure(figsize=(7, 4))
    plt.hist(u["eer"], bins=15, edgecolor="black")
    plt.title(f"Distribution of per-user EER ({best})")
    plt.xlabel("EER")
    plt.ylabel("Number of users")
    plt.tight_layout()
    plt.savefig(RESULTS / "user_eer_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    features = get_feature_cols(df)
    feature_analysis(df, features)
    training_size_curve(df, features)
    user_difficulty()
    print("\nSaved analysis to results/")
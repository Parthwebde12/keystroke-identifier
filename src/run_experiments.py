import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data import ROOT, get_feature_cols, load_data, split_by_session
from evaluate import compute_auc, compute_eer
from models import ALL_MODELS, score_model

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    df = load_data()
    features = get_feature_cols(df)
    train, test = split_by_session(df)
    subjects = sorted(df["subject"].unique())
    rng = np.random.default_rng(42)

    rows = []
    for i, user in enumerate(subjects, 1):
        # Split the other subjects into disjoint train/test impostor groups
        others = [s for s in subjects if s != user]
        rng.shuffle(others)
        half = len(others) // 2
        imp_train_subj, imp_test_subj = others[:half], others[half:]

        X_gen_train = train.loc[train["subject"] == user, features].values
        X_gen_test = test.loc[test["subject"] == user, features].values

        imp_train = (
            train[train["subject"].isin(imp_train_subj)]
            .groupby("subject")
            .sample(n=8, random_state=0)
        )
        imp_test = (
            test[test["subject"].isin(imp_test_subj)]
            .groupby("subject")
            .sample(n=20, random_state=0)
        )
        X_imp_train = imp_train[features].values
        X_imp_test = imp_test[features].values

        X_test = np.vstack([X_gen_test, X_imp_test])
        y_test = np.r_[np.ones(len(X_gen_test)), np.zeros(len(X_imp_test))]

        for name in ALL_MODELS:
            scores = score_model(name, X_gen_train, X_imp_train, X_test)
            rows.append(
                {
                    "user": user,
                    "model": name,
                    "eer": compute_eer(y_test, scores),
                    "auc": compute_auc(y_test, scores),
                }
            )
        print(f"[{i}/{len(subjects)}] finished {user}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "per_user_results.csv", index=False)

    summary = (
        results.groupby("model")[["eer", "auc"]]
        .agg(["mean", "median"])
        .round(4)
    )
    summary.columns = ["eer_mean", "eer_median", "auc_mean", "auc_median"]
    summary = summary.sort_values("eer_mean")
    summary.to_csv(RESULTS / "model_comparison.csv")
    print("\n=== Model comparison (lower EER is better) ===")
    print(summary)

    order = summary.index.tolist()
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=results, x="model", y="eer", order=order)
    plt.title("Per-user EER by model (lower is better)")
    plt.ylabel("Equal Error Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULTS / "eer_by_model.png", dpi=150)
    plt.close()
    print("\nSaved results to results/")


if __name__ == "__main__":
    main()
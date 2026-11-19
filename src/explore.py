import matplotlib.pyplot as plt
import seaborn as sns

from data import ROOT, get_feature_cols, load_data, split_by_session

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

df = load_data()
features = get_feature_cols(df)

print("Shape:", df.shape)
print("Subjects:", df["subject"].nunique())
print("Sessions:", sorted(df["sessionIndex"].unique()))
print("Feature count:", len(features))
print("Missing values:", int(df[features].isna().sum().sum()))
print("\nSamples per subject (first 5):")
print(df["subject"].value_counts().head())

train, test = split_by_session(df)
print(f"\nTrain rows: {len(train)} | Test rows: {len(test)}")

# Plot 1: one hold-time feature across a few subjects
hold_feature = [c for c in features if c.startswith("H.")][0]
subjects = sorted(df["subject"].unique())[:6]
plt.figure(figsize=(9, 5))
sns.boxplot(data=df[df["subject"].isin(subjects)], x="subject", y=hold_feature)
plt.title(f"{hold_feature} for 6 subjects: people differ")
plt.tight_layout()
plt.savefig(RESULTS / "hold_time_by_subject.png", dpi=150)
plt.close()

# Plot 2: how much a single user's mean typing speed drifts across sessions
one = df[df["subject"] == subjects[0]]
drift = one.groupby("sessionIndex")[hold_feature].mean()
plt.figure(figsize=(7, 4))
drift.plot(marker="o")
plt.title(f"{subjects[0]}: mean {hold_feature} per session")
plt.xlabel("Session")
plt.ylabel("Mean hold time (s)")
plt.tight_layout()
plt.savefig(RESULTS / "session_drift.png", dpi=150)
plt.close()

print("\nSaved plots to results/")
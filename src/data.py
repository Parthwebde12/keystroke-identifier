from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "DSL-StrongPasswordData.csv"  # change if your filename differs
META_COLS = ["subject", "sessionIndex", "rep"]


def load_data(path=DATA_PATH):
    """Load the CMU keystroke dataset and run basic sanity checks."""
    df = pd.read_csv(path)
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns not found: {missing}")
    return df


def get_feature_cols(df):
    """Return the 31 timing feature columns (H.*, DD.*, UD.*)."""
    return [c for c in df.columns if c not in META_COLS]


def split_by_session(df, train_sessions=(1, 2, 3, 4)):
    """Split by session so train and test never share a typing session."""
    is_train = df["sessionIndex"].isin(train_sessions)
    return df[is_train].copy(), df[~is_train].copy()
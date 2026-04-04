import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_df, v_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["city"])
    return t_df, v_df


def main() -> None:
    df = pd.read_csv("metadata_tables/full_data.csv")
    train_df, val_df = train_val_split(df)
    train_df.to_csv("metadata_tables/train_data.csv", index=False)
    val_df.to_csv("metadata_tables/val_data.csv", index=False)

if __name__ == "__main__":
    main()

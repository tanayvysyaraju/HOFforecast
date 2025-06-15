import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- STEP 1: Load and clean HOF + non-HOF datasets ---
non_hof_df = pd.read_csv("data/ML_nonHOF.csv")
hof_df = pd.read_csv("data/ML_HOF.csv")

# Rename columns in HOF to match non-HOF
rename_map = {
    "player": "Name", "position": "Position",
    "tackles": "Total Tackles", "sacks": "Sacks", "interception": "Ints",
    "receptions": "Receptions", "receiving_yards": "Receiving Yards", "receiving_td": "Receiving TDs",
    "rush_yards": "Rushing Yards", "rush_td": "Rushing TDs",
    "pass_yards": "Passing Yards", "pass_td": "TD Passes", "int_thrown": "Ints",
    "pass_attempts": "Passes Attempted", "completions": "Passes Completed"
}
hof_df.rename(columns=rename_map, inplace=True)

# Add HOF labels
hof_df["HOF"] = 1
non_hof_df["HOF"] = 0

# Ensure consistent columns across both
all_columns = set(hof_df.columns).union(set(non_hof_df.columns))
for col in all_columns:
    if col not in hof_df:
        hof_df[col] = pd.NA
    if col not in non_hof_df:
        non_hof_df[col] = pd.NA

# Drop duplicate columns if any
hof_df = hof_df.loc[:, ~hof_df.columns.duplicated()]
non_hof_df = non_hof_df.loc[:, ~non_hof_df.columns.duplicated()]

# Sort and remove duplicates while preserving order
unique_columns = []
seen = set()
for col in list(hof_df.columns) + list(non_hof_df.columns):
    if col not in seen:
        unique_columns.append(col)
        seen.add(col)

# Align DataFrames and concatenate
hof_aligned = hof_df.reindex(columns=unique_columns)
non_hof_aligned = non_hof_df.reindex(columns=unique_columns)
combined_df = pd.concat([hof_aligned, non_hof_aligned], ignore_index=True)

# --- STEP 2: Define position-specific relevant features ---
position_feature_map = {
    "QB": ["Passes Attempted", "Passes Completed", "Passing Yards", "TD Passes", "Ints"],
    "WR": ["Receptions", "Receiving Yards", "Receiving TDs"],
    "TE": ["Receptions", "Receiving Yards", "Receiving TDs"],
    "RB": ["Rushing Yards", "Rushing TDs", "Receptions", "Receiving Yards"],
    "LB": ["Total Tackles", "Sacks", "Ints"],
    "DE": ["Sacks", "Total Tackles"],
    "DT": ["Sacks", "Total Tackles"],
    "DB": ["Ints", "Total Tackles", "Passes Defended"],
    "SS": ["Ints", "Total Tackles", "Passes Defended"],
    "FS": ["Ints", "Total Tackles", "Passes Defended"],
}

# --- STEP 3: Train and evaluate models by position ---
positions = combined_df['Position'].dropna().unique()
for position in positions:
    print(f"\n=== Training for Position: {position} ===")
    features = position_feature_map.get(position)
    if not features:
        print(f"Skipping {position} — no relevant features defined.")
        continue

    pos_df = combined_df[combined_df['Position'] == position]
    if pos_df['HOF'].nunique() < 2 or len(pos_df) < 10:
        print(f"Skipping {position} — insufficient class diversity or samples.")
        continue

    # Prepare data
    X = pos_df[features].fillna(0)
    y = pos_df["HOF"]

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Feature Importances:")
    for feat, importance in zip(X.columns, model.feature_importances_):
        print(f"  {feat}: {importance:.4f}")

    # Optional: Save model or output predictions
    # import joblib
    # joblib.dump(model, f"models/{position}_rf_model.pkl")
